#ifndef TRACKER_H
#define TRACKER_H

#include "track.h"

#include <QMap>
#include <QTimer>
#include <boost/log/trivial.hpp>

class Tracker : public QObject
{
    Q_OBJECT

public:
    Tracker();
    ~Tracker();

    QMap<long long, Track*>* getTracks();
    void processNewTrack(double distance, double bearing,
                         double heading, double speed, double altitude,
                         Track::TrackType type, Track::TrackBehavior behavior);
    void removeTrack(Track *track);
    void removeTrack(long long track_id);

private slots:
    void periodicCheckTracks();

private:
    void initializeTracker();
    Track *getExistentTrack(double distance, double bearing, double heading,
                            double speed);

    long long getNextTrackId();
    void setTrackBehavior(Track *track);
    bool isSimilar(const Track *track, double distance,
                                 double bearing, double heading, double speed);

    QMap<long long, Track*> *m_tracklist;
    double m_distance_threshold;
    double m_bearing_threshold;
    double m_heading_threshold;
    double m_speed_threshold;

    QTimer *m_check_track_timer;

    boost::log::sources::severity_logger<boost::log::trivial::severity_level> m_logger;
};

#endif // TRACKER_H
