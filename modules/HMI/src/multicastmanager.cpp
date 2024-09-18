#include "multicastmanager.h"
#include "trackmessage.h"
#include "configurationmanager.h"

#include <QThread>

MulticastManager::MulticastManager(Tracker *tracker)
{
    m_tracker = tracker;

    m_thread.setObjectName("UdpThread");
    connect(&m_thread, &QThread::started, this, &MulticastManager::start);
}

void MulticastManager::start()
{
    int port = ConfigurationManager::getInstance()->getConnectionConfig().port;
    QString ip = QString::fromStdString(ConfigurationManager::getInstance()->getConnectionConfig().ip_address);
    m_socket.bind(QHostAddress::AnyIPv4, port, QUdpSocket::ShareAddress);
    m_socket.joinMulticastGroup(QHostAddress(ip));

    connect(&m_socket, &QUdpSocket::readyRead, this, &MulticastManager::readDatagram);
}

void MulticastManager::readDatagram()
{
    while (m_socket.hasPendingDatagrams()) {
        QByteArray datagram;
        datagram.resize(m_socket.pendingDatagramSize());
        QHostAddress sender;
        quint16 senderPort;

        m_socket.readDatagram(datagram.data(), datagram.size(), &sender, &senderPort);

        qDebug() << "Received multicast message from" << sender.toString() << "Port" << senderPort;
        qDebug() << "Message:" << datagram;

        TrackMessage message(datagram);
        message.unpack();

        Track::TrackType track_type = static_cast<Track::TrackType>(message.m_data.track_type);
        Track::TrackBehavior track_behavior = static_cast<Track::TrackBehavior>(message.m_data.behavior);
        m_tracker->processNewTrack(message.m_data.distance, message.m_data.bearing, message.m_data.heading, message.m_data.speed, 0, track_type, track_behavior);
    }
}
