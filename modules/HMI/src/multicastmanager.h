#ifndef MULTICASTMANAGER_H
#define MULTICASTMANAGER_H

#include "tracker.h"

#include <QObject>
#include <QtNetwork/QUdpSocket>
#include <QThread>

class MulticastManager : public QObject
{
    Q_OBJECT

public:
    MulticastManager(Tracker *tracker);

    QThread m_thread;

public slots:
    void start();
    void readDatagram();

private:
    QUdpSocket m_socket;
    Tracker *m_tracker;
};

#endif // MULTICASTMANAGER_H
