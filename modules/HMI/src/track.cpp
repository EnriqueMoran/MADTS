#include "track.h"

Track::Track()
{
    m_id = 0;
    m_distance = 0;
    m_bearing = 0;
    m_heading = 0;
    m_speed = 0;
    m_altitude = 0;
    m_timestamp = 0;
    m_type = TrackType::UNKNOWN;
    m_behavior = TrackBehavior::UNDEFINED;
}

Track::Track(const Track &other) :
    m_id(other.m_id),
    m_distance(other.m_distance),
    m_bearing(other.m_bearing),
    m_heading(other.m_heading),
    m_speed(other.m_speed),
    m_altitude(other.m_altitude),
    m_type(other.m_type),
    m_behavior(other.m_behavior)
{

}

long long Track::getId() const
{
    return m_id;
}

void Track::setId(long long newId)
{
    m_id = newId;
}

double Track::getBearing() const
{
    return m_bearing;
}

void Track::setBearing(double newBearing)
{
    m_bearing = newBearing;
}

double Track::getHeading() const
{
    return m_heading;
}

void Track::setHeading(double newHeading)
{
    m_heading = newHeading;
}

double Track::getSpeed() const
{
    return m_speed;
}

void Track::setSpeed(double newSpeed)
{
    m_speed = newSpeed;
}

double Track::getAltitude() const
{
    return m_altitude;
}

void Track::setAltitude(double newAltitude)
{
    m_altitude = newAltitude;
}

Track::TrackType Track::getType() const
{
    return m_type;
}

void Track::setType(TrackType newType)
{
    m_type = newType;
}

double Track::getDistance() const
{
    return m_distance;
}

void Track::setDistance(double newDistance)
{
    m_distance = newDistance;
}

Track::TrackBehavior Track::getBehavior() const
{
    return m_behavior;
}

void Track::setBehavior(TrackBehavior newBehavior)
{
    m_behavior = newBehavior;
}

QString Track::getBehaviorStr() const
{
    QString res = "";
    switch(m_behavior)
    {
    case TrackBehavior::ALLIED:
        res = "Ally";
        break;
    case TrackBehavior::HOSTILE:
        res = "Hostile";
        break;
    case TrackBehavior::UNDEFINED:
        res = "Undefined";
        break;
    default:
        res = "Unknown";
    }
    return res;
}

QString Track::getTypeStr() const
{
    QString res = "";
    switch(m_type)
    {
    case TrackType::AIR:
        res = "Air";
        break;
    case TrackType::GROUND:
        res = "Ground";
        break;
    case TrackType::SUBSURFACE:
        res = "Subsurface";
        break;
    case TrackType::SURFACE:
        res = "Surface";
        break;
    case TrackType::UNKNOWN:
        res = "Unknown";
        break;
    default:
        res = "Unknown";
    }
    return res;
}

long long Track::getTimestamp() const
{
    return m_timestamp;
}

void Track::setTimestamp(long long newTimestamp)
{
    m_timestamp = newTimestamp;
}
