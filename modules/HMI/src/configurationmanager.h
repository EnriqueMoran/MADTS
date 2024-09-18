#ifndef CONFIGURATIONMANAGER_H
#define CONFIGURATIONMANAGER_H

#include <iostream>
#include <QObject>
#include <boost/log/trivial.hpp>

class ConfigurationManager : public QObject
{
    Q_OBJECT

public:

    struct WindowConfig {
        std::string app_name;
        int window_width;
        int window_height;
        int view_window_width;
        int view_window_height;
        int menu_window_width;
        int menu_window_height;
        int view_background_r;
        int view_background_g;
        int view_background_b;
        int menu_background_r;
        int menu_background_g;
        int menu_background_b;
    };

    struct DrawConfig {
        int draw_interval;
        int max_tracks;
        double draw_range;
        double track_scale;
        double draw_step;
    };

    struct SliderConfig {
        int label_size;
        int label_spacing;
    };

    struct TrackerConfig {
        double distance_threshold;
        double bearing_threshold;
        double heading_threshold;
        double speed_threshold;
        long long track_timeout;
    };

    struct ConnectionConfig {
        std::string ip_address;
        int port;
        std::string interface;
    };

    static ConfigurationManager* getInstance();

    WindowConfig getWindowConfig();
    DrawConfig getDrawConfig();
    SliderConfig getSliderConfig();
    TrackerConfig getTrackerConfig();
    ConnectionConfig getConnectionConfig();

    void setRange(double new_range);


private:
    ConfigurationManager(QObject *parent = nullptr);
    void initialize();
    void readConfig(std::string file_path);
    static ConfigurationManager* instance;

    WindowConfig m_window_config;
    DrawConfig m_draw_config;
    SliderConfig m_slider_config;
    TrackerConfig m_tracker_config;
    ConnectionConfig m_connection_config;

    boost::log::sources::severity_logger<boost::log::trivial::severity_level> m_logger;
};

#endif // CONFIGURATIONMANAGER_H
