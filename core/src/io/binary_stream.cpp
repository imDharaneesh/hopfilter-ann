#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ios>
#include <span>
#include <string>

#include <hopfilter/common/errors.hpp>
#include <hopfilter/io/binary_stream.hpp>

namespace hopfilter {

void BinaryWriter::writeString(const std::string& text) {
    write(static_cast<std::uint32_t>(text.size()));
    writeAll(std::span<const char>{text});
}

void BinaryWriter::writeBytes(std::span<const std::byte> bytes) {
    std::string buffer(bytes.size(), '\0');
    std::memcpy(buffer.data(), bytes.data(), bytes.size());
    sink_->write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
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
    std::array<char, sizeof(std::uint32_t)> buffer{};
    source_->read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    if (source_->gcount() != static_cast<std::streamsize>(buffer.size())) {
        throw SerializationError("index data is truncated (missing checksum)");
    }
    std::uint32_t stored = 0;
    std::memcpy(&stored, buffer.data(), buffer.size());
    if (stored != expected) {
        throw SerializationError("index checksum mismatch; the data is corrupted");
    }
}

void BinaryReader::readBytes(std::span<std::byte> bytes) {
    std::string buffer(bytes.size(), '\0');
    source_->read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    if (source_->gcount() != static_cast<std::streamsize>(buffer.size())) {
        throw SerializationError("index data is truncated");
    }
    std::memcpy(bytes.data(), buffer.data(), buffer.size());
    crc_.update(bytes);
}

}
