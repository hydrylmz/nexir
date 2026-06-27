// src/io/slot_pool.rs

use std::sync::Mutex;

use crate::render::device::GpuDevice;

/// Opaque handle to one staging buffer slot in the pool.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct FrameSlotId {
    pub tier:  u8,   // which tier (0..3)
    pub index: u16,  // slot index within tier
}

impl FrameSlotId {
    pub fn index(&self) -> u16 { self.index }
}

/// One tier of identically-sized staging buffers.
struct SlotTier {
    slot_size:  u64,              // bytes per slot
    buffers:    Vec<std::sync::Mutex<Vec<u8>>>, // pre-allocated, one per slot
    free_list:  std::sync::Mutex<Vec<u16>>,  // indices of available slots
}

impl SlotTier {
    /// Pre-allocate `count` staging buffers of `raw_size` bytes each.
    pub fn new(device: &GpuDevice, raw_size: u64, count: u16, tier: u8) -> Self {
        let slot_size = raw_size; // No padding needed for Vec<u8>

        let mut buffers = Vec::with_capacity(count as usize);
        for _ in 0..count {
            let mut vec = Vec::with_capacity(slot_size as usize);
            vec.resize(slot_size as usize, 0);
            buffers.push(std::sync::Mutex::new(vec));
        }

        let free_list = std::sync::Mutex::new((0..count).collect());

        Self {
            slot_size,
            buffers,
            free_list,
        }
    }

    /// Acquire a free slot, or None if the tier is exhausted.
    pub fn acquire(&self, tier: u8) -> Option<FrameSlotId> {
        let index = self.free_list.lock().unwrap().pop()?;
        Some(FrameSlotId { tier, index })
    }

    /// Return a slot to the pool.
    pub fn release(&self, id: FrameSlotId) {
        self.free_list.lock().unwrap().push(id.index);
    }

    pub fn with_buffer_mut<F, R>(&self, index: u16, f: F) -> R
    where
        F: FnOnce(&mut [u8]) -> R,
    {
        let mut guard = self.buffers[index as usize].lock().unwrap();
        f(&mut guard)
    }

    pub fn with_buffer_read<F, R>(&self, index: u16, f: F) -> R
    where
        F: FnOnce(&[u8]) -> R,
    {
        let guard = self.buffers[index as usize].lock().unwrap();
        f(&guard)
    }
}

pub struct FrameSlotPool {
    tiers: [SlotTier; 4],
}

impl FrameSlotPool {
    /// Initialise the pool with default tier sizes and counts.
    pub fn new(device: &GpuDevice) -> Self {
        // Tier sizes:
        // Tier 0: <= 1920x1080 (YUV420p = 3.11MB)
        // Tier 1: <= 2560x1440 (YUV420p = 5.53MB)
        // Tier 2: <= 3840x2160 (YUV420p = 12.44MB)
        // Tier 3: <= 7680x4320 (YUV420p = 49.77MB)

        let tiers = [
            SlotTier::new(device, 3_110_400, 32, 0),
            SlotTier::new(device, 5_529_600, 16, 1),
            SlotTier::new(device, 12_441_600, 8, 2),
            SlotTier::new(device, 49_766_400, 2, 3),
        ];

        Self { tiers }
    }

    /// Find the appropriate tier for a given required byte size.
    fn tier_for(&self, required_bytes: u64) -> Option<usize> {
        for (i, tier) in self.tiers.iter().enumerate() {
            if required_bytes <= tier.slot_size {
                return Some(i);
            }
        }
        None
    }

    /// Acquire a staging buffer slot large enough for `required_bytes`.
    pub fn acquire(&self, required_bytes: u64) -> Option<FrameSlotId> {
        let tier_idx = self.tier_for(required_bytes)?;
        self.tiers[tier_idx].acquire(tier_idx as u8)
    }

    /// Release a slot back to its tier.
    pub fn release(&self, id: FrameSlotId) {
        self.tiers[id.tier as usize].release(id);
    }

    /// Get a reference to the staging buffer for a slot.
    pub fn with_buffer_mut<F, R>(&self, id: FrameSlotId, f: F) -> R
    where
        F: FnOnce(&mut [u8]) -> R,
    {
        self.tiers[id.tier as usize].with_buffer_mut(id.index, f)
    }

    pub fn with_buffer_read<F, R>(&self, id: FrameSlotId, f: F) -> R
    where
        F: FnOnce(&[u8]) -> R,
    {
        self.tiers[id.tier as usize].with_buffer_read(id.index, f)
    }
}
