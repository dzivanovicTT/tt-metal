
// Grid
uint8_t get_absolute_logical_x();
uint8_t get_absolute_logical_y();
uint8_t get_relative_logical_x();
uint8_t get_relative_logical_y();

// CB
uint32_t get_arg_addr(int arg_idx);
uint32_t get_common_arg_addr(int arg_idx);
template <typename T>
T get_arg_val(int arg_idx);
template <typename T>
T get_common_arg_val(int arg_idx);
void cb_push_back(const int32_t operand, const int32_t num_pages);
void cb_pop_front(int32_t operand, int32_t num_pages);
#ifdef DATA_FORMATS_DEFINED
constexpr std::int32_t get_tile_size(const std::int32_t operand);
constexpr uint32_t get_tile_hw(const std::int32_t operand);
constexpr uint32_t get_tile_num_faces(const std::int32_t operand);
constexpr DataFormat get_dataformat(const std::int32_t operand);
#endif
uint32_t get_write_ptr(uint32_t operand);
uint32_t get_read_ptr(uint32_t operand);
void wait_for_sync_register_value(uint32_t addr, int32_t val);
bool cb_pages_reservable_at_back(int32_t operand, int32_t num_pages);
void cb_reserve_back(int32_t operand, int32_t num_pages);
bool cb_pages_available_at_front(int32_t operand, int32_t num_pages);
void cb_wait_front(int32_t operand, int32_t num_pages);

// NOC
void noc_async_read_one_packet(
    std::uint64_t src_noc_addr, std::uint32_t dst_local_l1_addr, std::uint32_t size, uint8_t noc = noc_index);
template <uint32_t max_page_size>
void noc_async_read(std::uint64_t src_noc_addr, std::uint32_t dst_local_l1_addr, std::uint32_t size, uint8_t noc);
void noc_async_read_one_packet_set_state(std::uint64_t src_noc_addr, std::uint32_t size, uint8_t noc = noc_index);
template <bool inc_num_issued = true>
void noc_async_read_one_packet_with_state(
    std::uint32_t src_noc_addr, std::uint32_t dst_local_l1_addr, uint8_t noc = noc_index);
void noc_async_read_set_state(std::uint64_t src_noc_addr, uint8_t noc = noc_index);
template <bool inc_num_issued = true>
void noc_async_read_with_state(
    std::uint32_t src_noc_addr, std::uint32_t dst_local_l1_addr, std::uint32_t size, uint8_t noc = noc_index);
void noc_async_read_inc_num_issued(std::uint32_t num_issued_reads_inc, uint8_t noc = noc_index);
void noc_async_write_one_packet(
    std::uint32_t src_local_l1_addr, std::uint64_t dst_noc_addr, std::uint32_t size, uint8_t noc = noc_index);
void noc_async_write_multicast_one_packet(
    std::uint32_t src_local_l1_addr,
    std::uint64_t dst_noc_addr_multicast,
    std::uint32_t size,
    std::uint32_t num_dests,
    bool linked = false,
    bool multicast_path_reserve = true,
    uint8_t noc = noc_index);
template <bool non_posted = true>
void noc_async_write_one_packet_set_state(
    std::uint64_t dst_noc_addr, std::uint32_t size, uint8_t noc = noc_index, uint8_t vc = NOC_UNICAST_WRITE_VC);
template <bool non_posted = true>
void noc_async_write_one_packet_with_state(
    std::uint32_t src_local_l1_addr, std::uint32_t dst_noc_addr, uint8_t noc = noc_index);

template <bool DRAM>
void noc_async_read_page(
    const uint32_t id,
    const InterleavedAddrGen<DRAM>& s,
    std::uint32_t dst_local_l1_addr,
    uint32_t offset = 0,
    uint8_t noc = noc_index);

template <bool DRAM, uint32_t tile_hw>
void noc_async_read_tile(
    const uint32_t id,
    const InterleavedAddrGenFast<DRAM, tile_hw>& s,
    std::uint32_t dst_local_l1_addr,
    uint32_t offset = 0,
    uint8_t noc = noc_index);
template <uint32_t max_page_size = NOC_MAX_BURST_SIZE + 1>
void noc_async_write(
    std::uint32_t src_local_l1_addr, std::uint64_t dst_noc_addr, std::uint32_t size, uint8_t noc = noc_index);

template <bool DRAM, uint32_t tile_hw>
void noc_async_write_tile(
    const uint32_t id,
    const InterleavedAddrGenFast<DRAM, tile_hw>& s,
    std::uint32_t src_local_l1_addr,
    uint8_t noc = noc_index);

template <ProgrammableCoreType type = ProgrammableCoreType::TENSIX>
uint32_t get_semaphore(uint32_t semaphore_id);

void noc_semaphore_set_remote(std::uint32_t src_local_l1_addr, std::uint64_t dst_noc_addr, uint8_t noc = noc_index);
template <uint32_t max_page_size = NOC_MAX_BURST_SIZE + 1>
void noc_async_write_multicast(
    std::uint32_t src_local_l1_addr,
    std::uint64_t dst_noc_addr_multicast,
    std::uint32_t size,
    std::uint32_t num_dests,
    bool linked = false,
    bool multicast_path_reserve = true,
    uint8_t noc = noc_index);
void noc_semaphore_set_multicast(
    std::uint32_t src_local_l1_addr,
    std::uint64_t dst_noc_addr_multicast,
    std::uint32_t num_dests,
    bool linked = false,
    bool multicast_path_reserve = true,
    uint8_t noc = noc_index);
void noc_semaphore_set_multicast_loopback_src(
    std::uint32_t src_local_l1_addr,
    std::uint64_t dst_noc_addr_multicast,
    std::uint32_t num_dests,
    bool linked = false,
    bool multicast_path_reserve = true,
    uint8_t noc = noc_index);
void noc_async_write_multicast_loopback_src(
    std::uint32_t src_local_l1_addr,
    std::uint64_t dst_noc_addr_multicast,
    std::uint32_t size,
    std::uint32_t num_dests,
    bool linked = false,
    bool multicast_path_reserve = true,
    uint8_t noc = noc_index);
#ifdef ARCH_BLACKHOLE
void noc_async_write_multicast_exclude_region(
    std::uint32_t src_local_l1_addr,
    std::uint64_t dst_noc_addr_multicast,
    std::uint32_t size,
    std::uint32_t num_dests,
    std::uint32_t exclude_region,
    bool linked = false,
    bool multicast_path_reserve = true,
    uint8_t noc = noc_index);
#endif

void noc_async_read_barrier(uint8_t noc = noc_index);
void noc_async_write_barrier(uint8_t noc = noc_index);
void noc_async_writes_flushed(uint8_t noc = noc_index);
void noc_async_posted_writes_flushed(uint8_t noc = noc_index);
void noc_async_atomic_barrier(uint8_t noc_idx = noc_index);
void noc_async_full_barrier(uint8_t noc_idx = noc_index);
void noc_semaphore_wait(volatile tt_l1_ptr uint32_t* sem_addr, uint32_t val);
void noc_semaphore_wait_min(volatile tt_l1_ptr uint32_t* sem_addr, uint32_t val);
void noc_semaphore_set(volatile tt_l1_ptr uint32_t* sem_addr, uint32_t val);
template <bool write_to_stream_reg = false, bool posted = false>
void noc_inline_dw_write(uint64_t addr, uint32_t val, uint8_t be = 0xF, uint8_t noc = noc_index);
template <bool posted = false>
void noc_inline_dw_write_set_state(
    uint64_t addr, uint8_t be = 0xF, uint8_t cmd_buf = write_at_cmd_buf, uint8_t noc = noc_index);
template <bool update_addr_lo = false, bool update_counter = true, bool posted = false, bool update_addr_hi = false>
void noc_inline_dw_write_with_state(
    uint32_t val, uint32_t addr = 0, uint8_t cmd_buf = write_at_cmd_buf, uint8_t noc = noc_index);
template <bool posted = false>
void noc_semaphore_inc(uint64_t addr, uint32_t incr, uint8_t noc_id = noc_index);

void RISC_POST_HEARTBEAT(uint32_t& heartbeat);
uint32_t min(uint32_t a, uint32_t b);

template <bool use_vc>
uint32_t noc_async_read_tile_dram_sharded_set_state(
    uint32_t bank_base_address,
    uint32_t page_size,
    uint32_t bank_id = 0,
    const uint32_t vc = 0,
    uint8_t noc = noc_index);

void noc_async_read_tile_dram_sharded_with_state(
    uint32_t src_base_addr, uint32_t src_addr, uint32_t dest_addr, uint32_t trid = 0, uint8_t noc = noc_index);
template <bool skip_ptr_update = false>
void noc_async_read_tile_dram_sharded_with_state_with_trid(
    uint32_t src_base_addr, uint32_t src_addr, uint32_t dest_addr, uint32_t trid = 0, uint8_t noc = noc_index);
void noc_async_read_tile_dram_sharded_set_trid(uint32_t trid = 0, uint8_t noc = noc_index);
void noc_async_read_barrier_with_trid(uint32_t trid, uint8_t noc = noc_index);
void noc_async_write_one_packet_with_trid_set_state(
    std::uint64_t dst_noc_addr, uint8_t cmd_buf = write_cmd_buf, uint8_t noc = noc_index);
template <bool update_counter = true, bool posted = false>
void noc_async_write_one_packet_with_trid_with_state(
    std::uint32_t src_local_l1_addr,
    std::uint32_t dst_noc_addr,
    std::uint32_t size,
    std::uint32_t trid,
    uint8_t cmd_buf = write_cmd_buf,
    uint8_t noc = noc_index);
template <bool update_counter = true, bool posted = false>
void noc_async_write_one_packet_with_trid(
    std::uint32_t src_local_l1_addr,
    std::uint64_t dst_noc_addr,
    std::uint32_t size,
    std::uint32_t trid,
    uint8_t cmd_buf = write_cmd_buf,
    uint8_t noc = noc_index);

void noc_async_write_barrier_with_trid(uint32_t trid, uint8_t noc = noc_index);
