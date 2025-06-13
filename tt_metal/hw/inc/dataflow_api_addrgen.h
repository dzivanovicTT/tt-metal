namespace interleaved_addr_gen {

template <bool DRAM>
uint32_t get_bank_offset_index(uint32_t id);

template <bool DRAM>
uint32_t get_bank_index(uint32_t id, uint32_t bank_offset_index);

template <bool DRAM>
uint32_t get_noc_xy(uint32_t bank_index, uint8_t noc = noc_index);

template <bool DRAM>
uint32_t get_bank_offset(uint32_t bank_index);

template <bool DRAM>
constexpr uint32_t get_allocator_alignment();

template <bool DRAM>
constexpr uint32_t get_log_base2_of_allocator_alignment();
}  // namespace interleaved_addr_gen

template <uint32_t tile_hw = 1024>
constexpr static std::uint32_t MUL_WITH_TILE_SIZE(uint format, uint index);

std::uint64_t get_noc_multicast_addr(
    std::uint32_t noc_x_start,
    std::uint32_t noc_y_start,
    std::uint32_t noc_x_end,
    std::uint32_t noc_y_end,
    std::uint32_t addr,
    uint8_t noc = noc_index);

std::uint64_t get_noc_addr(std::uint32_t noc_x, std::uint32_t noc_y, std::uint32_t addr, uint8_t noc = noc_index);
std::uint64_t get_noc_addr_helper(std::uint32_t noc_xy, std::uint32_t addr);
std::uint32_t get_noc_exclude_region(
    std::uint32_t exclude_start_x,
    std::uint32_t exclude_start_y,
    std::uint32_t exclude_dir_x,
    std::uint32_t exclude_dir_y,
    uint8_t noc = noc_index);
uint64_t get_dram_noc_addr(
    const uint32_t id,
    const uint32_t page_size,
    const uint32_t bank_base_address,
    const uint32_t offset = 0,
    uint8_t noc = noc_index);
uint64_t get_l1_noc_addr(
    const uint32_t id,
    const uint32_t page_size,
    const uint32_t bank_base_address,
    const uint32_t offset = 0,
    uint8_t noc = noc_index);

uint64_t get_system_memory_noc_addr(
    const uint32_t id,
    const uint32_t page_size,
    const uint32_t base_addr,
    const uint32_t offset = 0,
    uint8_t noc = noc_index);

std::uint64_t get_noc_addr(std::uint32_t addr, uint8_t noc = noc_index);

template <bool DRAM>
struct InterleavedAddrGen {
    uint32_t get_addr(
        const uint32_t id,
        const uint32_t bank_offset_index,
        const uint32_t bank_index,
        const uint32_t offset = 0) const;

    std::uint64_t get_noc_addr(const uint32_t id, const uint32_t offset = 0, uint8_t noc = noc_index) const;

    void noc_async_read_page(
        const uint32_t id, const uint32_t dest_addr, const uint32_t offset = 0, uint8_t noc = noc_index) const;
};

template <bool DRAM>
struct InterleavedPow2AddrGen {
    uint32_t get_addr(
        const uint32_t id,
        const uint32_t bank_offset_index,
        const uint32_t bank_index,
        const uint32_t offset = 0) const;
    std::uint64_t get_noc_addr(const uint32_t id, const uint32_t offset = 0, uint8_t noc = noc_index) const;
};

template <bool DRAM, uint32_t tile_hw = 1024>
struct InterleavedAddrGenFast {
    uint32_t bank_base_address;  // Base address for the whole tensor.
    // TODO: Remove page_size from argument list. This can be derived from data_format
    uint32_t page_size;      // Num bytes in bank unit.
    DataFormat data_format;  // Data format

    uint32_t get_addr(
        const uint32_t id,
        const uint32_t bank_offset_index,
        const uint32_t bank_index,
        const uint32_t offset = 0) const;

    std::uint64_t get_noc_addr(const uint32_t id, const uint32_t offset = 0, uint8_t noc = noc_index) const;
    void noc_async_read_tile(
        const uint32_t id, uint32_t dest_addr, const uint32_t offset = 0, uint8_t noc = noc_index) const;
    void noc_async_write_tile(const uint32_t id, uint32_t src_addr, uint8_t noc = noc_index) const;
};

template <bool DRAM>
struct InterleavedPow2AddrGenFast {
    uint32_t get_addr(
        const uint32_t id,
        const uint32_t bank_offset_index,
        const uint32_t bank_index,
        const uint32_t offset = 0) const;
    std::uint64_t get_noc_addr(const uint32_t id, const uint32_t offset = 0, uint8_t noc = noc_index) const;
    void noc_async_read_page(
        const uint32_t id, uint32_t dest_addr, const uint32_t offset = 0, uint8_t noc = noc_index) const;
    void noc_async_read_partial_page(
        const uint32_t id,
        uint32_t dest_addr,
        const uint32_t size,
        const uint32_t offset,
        uint8_t noc = noc_index) const;
    void noc_async_write_page(
        const uint32_t id,
        uint32_t src_addr,
        const uint32_t write_size_bytes,
        const uint32_t offset = 0,
        uint8_t noc = noc_index) const;
};

template <bool DRAM>
std::uint64_t get_noc_addr(
    const uint32_t id, const InterleavedAddrGen<DRAM>& s, uint32_t offset = 0, uint8_t noc = noc_index);
template <bool DRAM>
std::uint64_t get_noc_addr(
    const uint32_t id, const InterleavedPow2AddrGen<DRAM>& s, uint32_t offset = 0, uint8_t noc = noc_index);
template <bool DRAM, uint32_t tile_hw>
std::uint64_t get_noc_addr(
    const uint32_t id, const InterleavedAddrGenFast<DRAM, tile_hw>& s, uint32_t offset = 0, uint8_t noc = noc_index);
template <bool DRAM>
std::uint64_t get_noc_addr(
    const uint32_t id, const InterleavedPow2AddrGenFast<DRAM>& s, uint32_t offset = 0, uint8_t noc = noc_index);
template <bool DRAM>
uint64_t get_noc_addr_from_bank_id(uint32_t bank_id, uint32_t bank_address_offset, uint8_t noc = noc_index);
template <bool DRAM, uint32_t page_size>
auto get_interleaved_addr_gen(uint32_t base_addr);
template <bool DRAM, bool is_size_pow2>
auto get_interleaved_addr_gen(uint32_t base_addr, uint32_t page_size, uint32_t log2_page_size);
template <bool DRAM, bool is_size_pow2>
auto get_interleaved_addr_gen(uint32_t base_addr, uint32_t size);
