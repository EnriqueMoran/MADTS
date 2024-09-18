#include "trackmessage.h"
#include <iostream>
#include <iomanip>
#include <cstring>

#ifdef _WIN32
    #include <winsock2.h>
    #include <ws2tcpip.h>
#else
    #include <arpa/inet.h>
#endif

#include <bitset>

using namespace boost::log::trivial;

TrackMessage::TrackMessage(QByteArray data)
{
#ifdef _WIN32
    WSADATA wsaData;
    int wsaResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (wsaResult != 0) {
        std::cerr << "WSAStartup failed with error: " << wsaResult << std::endl;
    }
#endif
    m_buffer = data;
}

TrackMessage::~TrackMessage()
{
#ifdef _WIN32
    WSACleanup();
#endif
}

void TrackMessage::pack()
{

}

void TrackMessage::unpack()
{
    BOOST_LOG_SEV(m_logger, debug) << "Unpacking data: ";

    std::string data_bits = byteArrayBitstoString(m_buffer);
    std::string data_bytes = byteArrayBytestoString(m_buffer);
    BOOST_LOG_SEV(m_logger, debug) << "    bits: " << data_bits;
    BOOST_LOG_SEV(m_logger, debug) << "    bytes: " << data_bytes;

    // Prototype
    uint8_t message_type;
    std::memcpy(&message_type, m_buffer.data(), sizeof(uint8_t));
    uint8_t id;
    std::memcpy(&id, m_buffer.data() + sizeof(uint8_t), sizeof(uint8_t));

    std::memcpy(&m_data.distance, m_buffer.data() + 2 * sizeof(uint8_t), sizeof(float));  // Little-endian float
    std::memcpy(&m_data.bearing, m_buffer.data() + 2 * sizeof(uint8_t) + sizeof(float), sizeof(int));  // Big-endian int
    std::memcpy(&m_data.heading, m_buffer.data() + 2 * sizeof(uint8_t) + sizeof(float) + sizeof(int), sizeof(int));  // Big-endian int
    std::memcpy(&m_data.speed, m_buffer.data() + 2 * sizeof(uint8_t) + sizeof(float) + 2 * sizeof(int), sizeof(float));  // Little-endian float
    std::memcpy(&m_data.track_type, m_buffer.data() + 2 * sizeof(uint8_t) + 2 * sizeof(float) + 2 * sizeof(int), sizeof(int));  // Big-endian int
    std::memcpy(&m_data.behavior, m_buffer.data() + 2 * sizeof(uint8_t) + 2 * sizeof(float) + 3 * sizeof(int), sizeof(int));  // Big-endian int

    m_data.bearing = ntohl(m_data.bearing);
    m_data.heading = ntohl(m_data.heading);
    m_data.track_type = ntohl(m_data.track_type);
    m_data.behavior = ntohl(m_data.behavior);

    BOOST_LOG_SEV(m_logger, debug) << "    Distance: " << m_data.distance;
    BOOST_LOG_SEV(m_logger, debug) << "    Bearing: " << m_data.bearing;
    BOOST_LOG_SEV(m_logger, debug) << "    Heading: " << m_data.heading;
    BOOST_LOG_SEV(m_logger, debug) << "    Speed: " << m_data.speed;
    BOOST_LOG_SEV(m_logger, debug) << "    Type: " << m_data.track_type;
    BOOST_LOG_SEV(m_logger, debug) << "    Behavior: " << m_data.behavior;
}

std::string TrackMessage::byteArrayBitstoString(const QByteArray &byteArray)
{
    std::string res;
    for (int i = 0; i < byteArray.size(); ++i) {
        uint8_t byte = static_cast<uint8_t>(byteArray[i]);
        std::bitset<8> bits(byte);

        res += bits.to_string();
    }
    return res;
}

std::string TrackMessage::byteArrayBytestoString(const QByteArray &byteArray)
{
    std::string res;
    for (int i = 0; i < byteArray.size(); ++i) {
        uint8_t byte = static_cast<uint8_t>(byteArray[i]);

        res += static_cast<char>(byte);
    }
    return res;
}
