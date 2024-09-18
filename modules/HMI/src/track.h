#ifndef TRACK_H
#define TRACK_H

#include <QString>

class Track
{
public:

    enum TrackType
    {
        AIR,
        GROUND,
        SUBSURFACE,
        SURFACE,
        UNKNOWN
    };

    enum TrackBehavior
    {
        ALLIED,
        HOSTILE,
        UNDEFINED
    };

    Track();

    Track(const Track& other);

    bool operator==(const Track& other) const {
        return m_id == other.m_id;
    }

    long long getId() const;
    void setId(long long newId);

    double getBearing() const;
    void setBearing(double newBearing);

    double getHeading() const;
    void setHeading(double newHeading);

    double getSpeed() const;
    void setSpeed(double newSpeed);

    double getAltitude() const;
    void setAltitude(double newAltitude);

    TrackType getType() const;
    void setType(TrackType newType);

    double getDistance() const;
    void setDistance(double newDistance);

    TrackBehavior getBehavior() const;
    void setBehavior(TrackBehavior newBehavior);

    QString getBehaviorStr() const;
    QString getTypeStr() const;

    long long getTimestamp() const;
    void setTimestamp(long long newTimestamp);

private:
    long long m_id;
    double m_distance;    // Distance to ownship
    double m_bearing;     // Bearing relative to ownship
    double m_heading;     // Heading relative to North
    double m_speed;       // Absolute speed
    double m_altitude;    // Elevation above surface
    long long m_timestamp;
    TrackType m_type;
    TrackBehavior m_behavior;
};

#endif // TRACK_H
