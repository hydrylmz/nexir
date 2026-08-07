use crate::export::renderer::RawFrame;
use std::sync::{mpsc, Mutex};

const QUEUE_DEPTH: usize = 8;

pub struct EncoderQueue {
    tx: mpsc::SyncSender<QueueItem>,
    rx: Mutex<mpsc::Receiver<QueueItem>>,
}

pub enum QueueItem {
    Frame(RawFrame),
    SegmentDone { segment_index: usize },
    AllDone,
}

impl EncoderQueue {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::sync_channel(QUEUE_DEPTH);
        Self { tx, rx: Mutex::new(rx) }
    }

    pub fn push(&self, item: QueueItem) {
        self.tx.send(item).expect("encoder queue receiver disconnected");
    }

    pub fn pop(&self) -> Option<QueueItem> {
        self.rx.lock().unwrap().recv().ok()
    }

    pub fn try_pop(&self) -> Option<QueueItem> {
        self.rx.lock().unwrap().try_recv().ok()
    }
}
