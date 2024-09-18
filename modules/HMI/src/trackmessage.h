#ifndef TRACKMESSAGE_H
#define TRACKMESSAGE_H

#include <QByteArray>
#include <boost/log/trivial.hpp>

class TrackMessage
{
    struct Data {
        float distance;
        int bearing;
        int heading;
        float speed;
        int track_type;
        int behavior;
    };

public:
    TrackMessage(QByteArray data);
    ~TrackMessage();

    void pack();
    void unpack();

    Data m_data;

private:
    std::string byteArrayBitstoString(const QByteArray& byteArray);
    std::string byteArrayBytestoString(const QByteArray& byteArray);

    QByteArray m_buffer;

    boost::log::sources::severity_logger<boost::log::trivial::severity_level> m_logger;
};

#endif // TRACKMESSAGE_H
