#include "tracker.h"
#include "configurationmanager.h"

#include <chrono>
#include <QDateTime>
#include <QRandomGenerator>

using namespace boost::log::trivial;

Tracker::Tracker()
{
    m_tracklist = new QMap<long long, Track*>();
    m_distance_threshold = ConfigurationManager::getInstance()->getTrackerConfig().distance_threshold;
    m_bearing_threshold = ConfigurationManager::getInstance()->getTrackerConfig().bearing_threshold;
    m_heading_threshold = ConfigurationManager::getInstance()->getTrackerConfig().heading_threshold;
    m_speed_threshold = ConfigurationManager::getInstance()->getTrackerConfig().speed_threshold;

    m_check_track_timer = new QTimer(this);
    connect(m_check_track_timer, SIGNAL(timeout()), this, SLOT(periodicCheckTracks()));
    m_check_track_timer->start(1000);

    /*
                                OFFLINE TEST DATA
    processNewTrack(1, 0, 90, 54, 0, Track::TrackType::AIR, Track::TrackBehavior::HOSTILE);
    processNewTrack(1, 0, 90, 51, 0, Track::TrackType::AIR, Track::TrackBehavior::ALLIED);
    processNewTrack(1, 45, 341, 66, 0, Track::TrackType::AIR, Track::TrackBehavior::UNDEFINED);
    processNewTrack(1, 90, 341, 66, 0, Track::TrackType::AIR, Track::TrackBehavior::UNDEFINED);
    processNewTrack(1, 135, 90, 54, 0, Track::TrackType::AIR, Track::TrackBehavior::UNDEFINED);
    processNewTrack(1, 180, 341, 66, 0, Track::TrackType::AIR, Track::TrackBehavior::ALLIED);
    processNewTrack(1, 225, 341, 66, 0, Track::TrackType::AIR, Track::TrackBehavior::ALLIED);
    processNewTrack(1, 270, 90, 54, 0, Track::TrackType::AIR, Track::TrackBehavior::ALLIED);
    processNewTrack(1, 315, 341, 66, 0, Track::TrackType::AIR, Track::TrackBehavior::ALLIED);
    */
}

Tracker::~Tracker()
{
    for (auto it = m_tracklist->begin(); it != m_tracklist->end(); ++it) {
        delete it.value();
    }

    m_tracklist->clear();
    delete m_tracklist;
}

QMap<long long, Track *> *Tracker::getTracks()
{
    return m_tracklist;
}

void Tracker::processNewTrack(double distance, double bearing, double heading,
                              double speed, double altitude, Track::TrackType type, Track::TrackBehavior behavior)
{
    BOOST_LOG_SEV(m_logger, info) << "Tracker - Received process new track: ";
    BOOST_LOG_SEV(m_logger, info) << "    distance: " << distance;
    BOOST_LOG_SEV(m_logger, info) << "    bearing: " << bearing;
    BOOST_LOG_SEV(m_logger, info) << "    heading: " << heading;
    BOOST_LOG_SEV(m_logger, info) << "    speed: " << speed;
    BOOST_LOG_SEV(m_logger, info) << "    altitude: " << altitude;
    BOOST_LOG_SEV(m_logger, info) << "    type: " << static_cast<int>(type);
    BOOST_LOG_SEV(m_logger, info) << "    behavior: " << static_cast<int>(behavior);

    // Check if there is an existent track in map
    Track *existent_track = getExistentTrack(distance, bearing, heading, speed);

    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    long long millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

    if (existent_track == nullptr)
    {
        // Track not found in map, create it
        long long new_track_id = getNextTrackId();
        if (new_track_id == -1)    // New id couldnt be assigned (there are already 50 tracks)
        {
            BOOST_LOG_SEV(m_logger, error) << "Tracker - New id couldnt be assigned to track, aborting!";
            return;
        }
        else
        {
            existent_track = new Track();
            existent_track->setId(new_track_id);
            existent_track->setBehavior(behavior);
            existent_track->setType(type);
            BOOST_LOG_SEV(m_logger, info) << "Tracker - Track not found. Assigned id: " << existent_track->getId();
        }
    }
    else
    {
        BOOST_LOG_SEV(m_logger, info) << "Tracker - Track found. Id: " << existent_track->getId();
    }
    existent_track->setDistance(distance);
    existent_track->setBearing(bearing);
    existent_track->setHeading(heading);
    existent_track->setSpeed(speed);
    existent_track->setAltitude(altitude);    
    existent_track->setTimestamp(millis);
    setTrackBehavior(existent_track);

    m_tracklist->insert(existent_track->getId(), existent_track);
    BOOST_LOG_SEV(m_logger, info) << "Tracker - Track data added to internal map.";
}


void Tracker::removeTrack(Track *track)
{
    m_tracklist->remove(track->getId());
}

void Tracker::removeTrack(long long track_id)
{
    m_tracklist->remove(track_id);
}

void Tracker::periodicCheckTracks()
{
    BOOST_LOG_SEV(m_logger, debug) << "Tracker - Checking tracks. Track list size: " << m_tracklist->keys().size();

    for (int i=0; i < m_tracklist->keys().size(); i++)
    {
        int track_id = m_tracklist->keys()[i];
        Track *track = m_tracklist->value(track_id);
        if (track)
        {
            BOOST_LOG_SEV(m_logger, debug) << "    Track id: " << track->getId();

            uint64_t current_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            QDateTime current_time_qt = QDateTime::fromMSecsSinceEpoch(current_time);
            QString current_time_str = current_time_qt.toString("yyyy-MM-dd hh:mm:ss");

            QDateTime track_time_qt = QDateTime::fromMSecsSinceEpoch(track->getTimestamp());
            QString track_time_str = track_time_qt.toString("yyyy-MM-dd hh:mm:ss");

            uint64_t track_timestamp = static_cast<uint64_t>(track->getTimestamp());
            uint64_t diff = current_time - track_timestamp;
            QDateTime diff_qt = QDateTime::fromMSecsSinceEpoch(diff);
            QString diff_str = diff_qt.toString("hh:mm:ss.zzz");

            BOOST_LOG_SEV(m_logger, debug) << "    Current timestamp: " << current_time << "(" << current_time_str.toStdString() << ") ";
            BOOST_LOG_SEV(m_logger, debug) << "    Track timestamp: " << track->getTimestamp() << "(" << track_time_str.toStdString() << ") ";
            BOOST_LOG_SEV(m_logger, debug) << "    Time difference: " << diff << "(" << diff_str.toStdString() << ")";

            if (diff > static_cast<uint64_t>(ConfigurationManager::getInstance()->getTrackerConfig().track_timeout))
            {
                BOOST_LOG_SEV(m_logger, debug) << "    Time difference is higher than track timeout (" <<
                    ConfigurationManager::getInstance()->getTrackerConfig().track_timeout << ")";

                m_tracklist->remove(track->getId());
                delete track;
                BOOST_LOG_SEV(m_logger, debug) << "    Track removed from internal list";
            }
        }
    }
}

void Tracker::initializeTracker()
{
    BOOST_LOG_SEV(m_logger, info) << "Tracker - Initializing tracker";
    for (int i=0; i < ConfigurationManager::getInstance()->getDrawConfig().max_tracks; i++)
    {
        m_tracklist->insert(i, nullptr);
    }
    BOOST_LOG_SEV(m_logger, info) << "    Tracker initialized, max tracks allowed: " << m_tracklist->keys().size();
}

Track *Tracker::getExistentTrack(double distance, double bearing, double heading,
                                 double speed)
{
    Track *found_track = nullptr;
    for (auto it = m_tracklist->begin(); it != m_tracklist->end(); ++it) {
        if (it.value() != nullptr && isSimilar(it.value(), distance, bearing, heading, speed)) {
            found_track = it.value();
            break;
        }
    }
    return found_track;
}

long long Tracker::getNextTrackId()
{
    long long new_id = -1;
    for (int i=0; i < ConfigurationManager::getInstance()->getDrawConfig().max_tracks; i++)
    {
        if (m_tracklist->value(i) == nullptr)
        {
            new_id = i;
            break;
        }
    }
    return new_id;
}

void Tracker::setTrackBehavior(Track *track)
{
    // TODO: Improve
    if (track->getBehavior() != Track::TrackBehavior::ALLIED)
    {
        if (abs(track->getBearing() - (int((track->getHeading() + 180)) % 360)) < 10 && track->getSpeed() > 15)
        {

            BOOST_LOG_SEV(m_logger, info) << "Tracker - Track with id: " << track->getId() << " set as Hostile";
            track->setBehavior(Track::TrackBehavior::HOSTILE);
        }
        else
        {
            BOOST_LOG_SEV(m_logger, info) << "Tracker - Track with id: " << track->getId() << " set as Undefined";
            track->setBehavior(Track::TrackBehavior::UNDEFINED);
        }
    }
}

bool Tracker::isSimilar(const Track *track, double distance,
                                      double bearing, double heading, double speed)
{
    BOOST_LOG_SEV(m_logger, info) << "Track distance: " << track->getDistance()
                                  << " bearing: " << std::abs(track->getBearing())
                                  << " heading: " << track->getHeading()
                                  << " speed: " << track->getSpeed();

    BOOST_LOG_SEV(m_logger, info) << "Comparing distance: " << distance
                                  << " bearing: " << bearing
                                  << " heading: " << heading
                                  << " speed: " << speed;

    double distance_diff = std::abs(track->getDistance() - distance);
    int bearing_diff = std::min(std::abs(track->getBearing() - bearing), 360 - std::abs(track->getBearing() - bearing));
    int heading_diff = std::min(std::abs(track->getHeading() - heading), 360 - std::abs(track->getHeading() - heading));
    int speed_diff = std::abs(track->getSpeed() - speed);

    BOOST_LOG_SEV(m_logger, info) << "Distance diff: " << distance_diff
                                  << " bearing diff: " << bearing_diff
                                  << " heading diff: " << heading_diff
                                  << " speed diff: " << speed_diff;

    return std::abs(track->getDistance() - distance) <= (m_distance_threshold) &&
           std::min(std::abs(track->getBearing() - bearing), 360 - std::abs(track->getBearing() - bearing)) <= m_bearing_threshold &&
           std::min(std::abs(track->getHeading() - heading), 360 - std::abs(track->getHeading() - heading)) <= m_heading_threshold &&
           std::abs(track->getSpeed() - speed) <= m_speed_threshold;
}
