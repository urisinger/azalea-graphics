use ash::vk;

pub struct TimestampQueryPool {
    pub handle: vk::QueryPool,
    pub count: u32,

    reset: bool,
}

impl TimestampQueryPool {
    pub fn new(device: &ash::Device, count: u32) -> Result<Self, vk::Result> {
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(count);

        let handle = unsafe { device.create_query_pool(&info, None)? };
        Ok(Self {
            handle,
            count,
            reset: false,
        })
    }

    pub fn reset(
        &mut self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        first_query: u32,
        query_count: u32,
    ) {
        self.reset = true;
        unsafe { device.cmd_reset_query_pool(cmd, self.handle, first_query, query_count) }
    }

    pub fn write_timestamp(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        query_index: u32,
        stage: vk::PipelineStageFlags,
    ) {
        unsafe { device.cmd_write_timestamp(cmd, stage, self.handle, query_index) }
    }

    pub fn get_results(&self, device: &ash::Device, results: &mut [u64]) {
        if self.reset {
            unsafe {
                device
                    .get_query_pool_results(
                        self.handle,
                        0,
                        results,
                        vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                    )
                    .unwrap();
            }
        }
    }

    pub fn destroy(&self, device: &ash::Device){
        unsafe{
            device.destroy_query_pool(self.handle, None);
        }

    }
}
