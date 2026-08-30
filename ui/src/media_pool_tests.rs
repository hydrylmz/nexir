// ui/src/media_pool_tests.rs
//
// P2.7 — media import, up to but not including the file dialog.
//
// THE SEAM. `rfd::FileDialog::pick_files` blocks on a human and panics without a
// display, so it is genuinely untestable here. Everything on the other side of it
// is not: what an imported path MEANS (video? audio? still image?), what happens
// when the same file is imported twice, and what a selection index refers to
// afterwards. That logic lives on `MediaPoolState::apply_pending_import`, which
// `draw` calls directly — so these tests exercise the shipped path rather than a
// re-implementation of it.
//
// WHY CLASSIFICATION IS WORTH PINNING. `MediaKind` decides the icon, but
// `is_still_image_path` decides something load-bearing: the scheduler quantises a
// still image's PTS to 0 and rewrites its frame rate to 0/1
// (`FrameScheduler::process_island_cached`), and `ProjectFile::from` does the same
// on load. A `.png` misclassified as video would be scheduled at a frame rate it
// does not have. The extension table here and `is_still_image_path` in the library
// are two separate lists that must agree, and nothing else checks that they do.

use crate::layout::media_pool::{MediaEntry, MediaKind, MediaPoolState};
use std::path::PathBuf;

/// Queue paths as though the file dialog had returned them, then apply.
fn import(state: &mut MediaPoolState, paths: &[&str]) -> usize {
    state.pending_import = Some(paths.iter().map(PathBuf::from).collect());
    state.apply_pending_import()
}

/// Every extension the import filter offers must classify, and match the
/// library's own still-image predicate.
///
/// The filter in `draw` advertises 19 extensions. Each is asserted against the
/// kind it should get, and the image cases are cross-checked against
/// `nexir::timeline::source::is_still_image_path` — the function the scheduler and
/// the project loader actually branch on. A disagreement between the UI's list and
/// that predicate is how a still image ends up scheduled as video.
#[test]
fn every_offered_extension_classifies_and_agrees_with_the_library() {
    use nexir::timeline::source::is_still_image_path;

    // (path, expected kind) — the full set from the dialog's filter.
    let cases: &[(&str, MediaKind)] = &[
        ("a.mp4",  MediaKind::Video),
        ("a.mov",  MediaKind::Video),
        ("a.mkv",  MediaKind::Video),
        ("a.avi",  MediaKind::Video),
        ("a.webm", MediaKind::Video),
        ("a.mxf",  MediaKind::Video),
        ("a.m4v",  MediaKind::Video),
        ("a.mp3",  MediaKind::Audio),
        ("a.wav",  MediaKind::Audio),
        ("a.aac",  MediaKind::Audio),
        ("a.flac", MediaKind::Audio),
        ("a.ogg",  MediaKind::Audio),
        ("a.m4a",  MediaKind::Audio),
        ("a.png",  MediaKind::Image),
        ("a.jpg",  MediaKind::Image),
        ("a.jpeg", MediaKind::Image),
        ("a.bmp",  MediaKind::Image),
        ("a.tiff", MediaKind::Image),
        ("a.webp", MediaKind::Image),
    ];

    for (path, want) in cases {
        let entry = MediaEntry::from_path(PathBuf::from(path));
        assert_eq!(
            entry.kind, *want,
            "{path} classified as {:?}, expected {want:?}",
            entry.kind
        );

        // The cross-check: the UI's notion of "image" must be the library's.
        let library_says_image = is_still_image_path(std::path::Path::new(path));
        assert_eq!(
            entry.kind == MediaKind::Image, library_says_image,
            "{path}: the media pool says image={} but \
             `is_still_image_path` says {library_says_image}. These two lists \
             disagreeing means a still is scheduled at a frame rate it does not \
             have, or a video is pinned to PTS 0.",
            entry.kind == MediaKind::Image
        );

        // Every entry keeps a displayable name, since the row renders it.
        assert!(!entry.name.is_empty(), "{path} produced an empty display name");
    }

    // Case must not matter: users have .MP4 and .PNG files.
    for (upper, want) in [("CLIP.MP4", MediaKind::Video), ("STILL.PNG", MediaKind::Image)] {
        let entry = MediaEntry::from_path(PathBuf::from(upper));
        assert_eq!(
            entry.kind, want,
            "{upper} classified as {:?} — extension matching is case-sensitive",
            entry.kind
        );
    }

    // An unknown extension falls back to Video rather than panicking or being
    // dropped: the user asked for it, and the demuxer is the real authority on
    // whether it opens.
    let unknown = MediaEntry::from_path(PathBuf::from("mystery.r3d"));
    assert_eq!(
        unknown.kind, MediaKind::Video,
        "an unrecognised extension must default to Video, not vanish"
    );
    // And a file with no extension at all must not panic.
    let bare = MediaEntry::from_path(PathBuf::from("no_extension"));
    assert_eq!(bare.name, "no_extension");
}

/// Re-importing a file the pool already holds must not duplicate it.
///
/// A second row for the same path gives the media two selectable entries and two
/// drag sources, and `selected: Option<usize>` then means something different
/// depending on which one the user clicked. De-duplication is by PATH, which is
/// also the key `StillImageCache` and `SourceRegistry::id_for_path` use.
#[test]
fn importing_the_same_file_twice_does_not_duplicate_it() {
    let mut state = MediaPoolState::default();

    assert_eq!(import(&mut state, &["a.mp4", "b.wav"]), 2);
    assert_eq!(state.entries.len(), 2);

    // The same two again, plus one new one.
    assert_eq!(
        import(&mut state, &["a.mp4", "b.wav", "c.png"]), 1,
        "a re-import reported adding more than the one genuinely new file"
    );
    assert_eq!(
        state.entries.len(), 3,
        "duplicate entries were added: {:?}",
        state.entries.iter().map(|e| &e.name).collect::<Vec<_>>()
    );

    // Duplicates WITHIN a single import batch are also collapsed — a user can
    // ctrl-click the same file twice in some dialogs, and `pick_files` returns
    // whatever it was given.
    let mut fresh = MediaPoolState::default();
    assert_eq!(
        import(&mut fresh, &["dup.mp4", "dup.mp4", "dup.mp4"]), 1,
        "the same path repeated inside one batch produced multiple entries"
    );
    assert_eq!(fresh.entries.len(), 1);

    // Import order is preserved: the pool is a list the user scans, and reordering
    // it on them is its own small bug.
    let mut ordered = MediaPoolState::default();
    import(&mut ordered, &["z.mp4", "a.mp4", "m.wav"]);
    assert_eq!(
        ordered.entries.iter().map(|e| e.name.as_str()).collect::<Vec<_>>(),
        vec!["z.mp4", "a.mp4", "m.wav"],
        "the entry list was reordered relative to the import"
    );
}

/// Applying with nothing queued is a no-op.
///
/// `draw` calls `apply_pending_import` unconditionally on every frame — sixty times
/// a second — so the empty case is by far the hottest, and it must neither allocate
/// nor disturb the existing list.
#[test]
fn applying_with_nothing_queued_changes_nothing() {
    let mut state = MediaPoolState::default();
    import(&mut state, &["a.mp4"]);
    state.selected = Some(0);

    let before: Vec<PathBuf> = state.entries.iter().map(|e| e.path.clone()).collect();

    for _ in 0..10 {
        assert_eq!(
            state.apply_pending_import(), 0,
            "an empty apply reported adding entries"
        );
    }

    assert_eq!(
        state.entries.iter().map(|e| e.path.clone()).collect::<Vec<_>>(),
        before,
        "repeated empty applies modified the entry list"
    );
    assert_eq!(
        state.selected, Some(0),
        "an empty apply disturbed the selection"
    );
    assert!(
        state.pending_import.is_none(),
        "pending_import was left set after being applied"
    );
}

/// An empty selection from the dialog is distinct from no dialog at all.
///
/// `pick_files` returns `None` when the user cancels, which `draw` never turns
/// into a `Some(vec![])` — but a future refactor could, and the result must still
/// be a clean no-op rather than an entry with an empty path.
#[test]
fn an_empty_selection_adds_nothing() {
    let mut state = MediaPoolState::default();
    import(&mut state, &["keep.mp4"]);

    state.pending_import = Some(Vec::new());
    assert_eq!(state.apply_pending_import(), 0);
    assert_eq!(
        state.entries.len(), 1,
        "an empty selection added an entry"
    );
    assert!(state.pending_import.is_none());
}
