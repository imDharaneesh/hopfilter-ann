#include <cstddef>
#include <cstdint>
#include <ios>
#include <span>
#include <string>

#include <hopfilter/common/errors.hpp>
#include <hopfilter/io/binary_stream.hpp>

namespace hopfilter {
namespace {

// char and std::byte may alias any object, which is exactly what stream I/O needs.
const char* asChars(const std::byte* bytes) noexcept {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return reinterpret_cast<const char*>(bytes);
}

char* asChars(std::byte* bytes) noexcept {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return reinterpret_cast<char*>(bytes);
}

}  // namespace

void BinaryWriter::writeString(const std::string& text) {
    write(static_cast<std::uint32_t>(text.size()));
    writeAll(std::span<const char>{text});
}

void BinaryWriter::writeBytes(std::span<const std::byte> bytes) {
    sink_->write(asChars(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!*sink_) {
        throw SerializationError("failed to write index data");
    }
    crc_.update(bytes);
}

std::string BinaryReader::readString(std::size_t maxLength) {
    const auto length = read<std::uint32_t>();
    if (length > maxLength) {
        throw SerializationError("string of length " + std::to_string(length) +
                                 " exceeds the limit of " + std::to_string(maxLength));
    }
    std::string text(length, '\0');
    readAll(std::span<char>{text});
    return text;
}

void BinaryReader::verifyChecksum() {
    const std::uint32_t expected = crc_.value();
    std::uint32_t stored = 0;
    source_->read(asChars(std::as_writable_bytes(std::span{&stored, 1}).data()), sizeof(stored));
    if (source_->gcount() != sizeof(stored)) {
        throw SerializationError("index data is truncated (missing checksum)");
    }
    if (stored != expected) {
        throw SerializationError("index checksum mismatch; the data is corrupted");
    }
}

void BinaryReader::readBytes(std::span<std::byte> bytes) {
    source_->read(asChars(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (source_->gcount() != static_cast<std::streamsize>(bytes.size())) {
        throw SerializationError("index data is truncated");
    }
    crc_.update(bytes);
}

}  // namespace hopfilter
