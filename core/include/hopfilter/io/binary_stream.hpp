#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
#include <span>
#include <string>
#include <type_traits>

#include <hopfilter/io/crc32.hpp>

namespace hopfilter {

static_assert(std::endian::native == std::endian::little,
              "hopfilter serialization currently supports little-endian hosts only");

class BinaryWriter {
public:
    explicit BinaryWriter(std::ostream& sink) noexcept : sink_(&sink) {}

    template <typename Value>
        requires std::is_trivially_copyable_v<Value>
    void write(const Value& value) {
        writeBytes(std::as_bytes(std::span<const Value, 1>{&value, 1}));
    }

    template <typename Value>
        requires std::is_trivially_copyable_v<Value>
    void writeAll(std::span<const Value> values) {
        writeBytes(std::as_bytes(values));
    }

    void writeString(const std::string& text);

    [[nodiscard]] std::uint32_t checksum() const noexcept { return crc_.value(); }

private:
    void writeBytes(std::span<const std::byte> bytes);

    std::ostream* sink_;
    Crc32 crc_;
};

class BinaryReader {
public:
    explicit BinaryReader(std::istream& source) noexcept : source_(&source) {}

    template <typename Value>
        requires std::is_trivially_copyable_v<Value>
    [[nodiscard]] Value read() {
        Value value{};
        readBytes(std::as_writable_bytes(std::span<Value, 1>{&value, 1}));
        return value;
    }

    template <typename Value>
        requires std::is_trivially_copyable_v<Value>
    void readAll(std::span<Value> values) {
        readBytes(std::as_writable_bytes(values));
    }

    [[nodiscard]] std::string readString(std::size_t maxLength);

    void verifyChecksum();

private:
    void readBytes(std::span<std::byte> bytes);

    std::istream* source_;
    Crc32 crc_;
};

}
