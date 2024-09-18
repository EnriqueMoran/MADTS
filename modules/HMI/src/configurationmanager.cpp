#include "configurationmanager.h"
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>

#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/sources/severity_logger.hpp>

using namespace boost::log::trivial;

ConfigurationManager* ConfigurationManager::instance = nullptr;

ConfigurationManager *ConfigurationManager::getInstance()
{
    if (!instance)
    {
        instance = new ConfigurationManager();
    }
    return instance;
}

ConfigurationManager::WindowConfig ConfigurationManager::getWindowConfig()
{
    return m_window_config;
}

ConfigurationManager::DrawConfig ConfigurationManager::getDrawConfig()
{
    return m_draw_config;
}

ConfigurationManager::SliderConfig ConfigurationManager::getSliderConfig()
{
    return m_slider_config;
}

ConfigurationManager::TrackerConfig ConfigurationManager::getTrackerConfig()
{
    return m_tracker_config;
}

ConfigurationManager::ConnectionConfig ConfigurationManager::getConnectionConfig()
{
    return m_connection_config;
}

void ConfigurationManager::setRange(double new_range)
{
    BOOST_LOG_SEV(m_logger, debug) << "Range changed from: " << m_draw_config.draw_range << " to: " << new_range;
    m_draw_config.draw_range = new_range;
}

ConfigurationManager::ConfigurationManager(QObject *parent)
{
    initialize();
}

void ConfigurationManager::initialize()
{
    std::string file_path = "./cfg/configuration.ini";
    readConfig(file_path);
}

void ConfigurationManager::readConfig(std::string file_path)
{
    BOOST_LOG_SEV(m_logger, info) << "Reading configuration from: " << file_path;
    try {
        boost::property_tree::ptree pt;
        boost::property_tree::ini_parser::read_ini(file_path, pt);

        m_window_config.app_name           = pt.get<std::string>("WINDOW.app_name", "HMI");
        m_window_config.window_width       = pt.get<int>("WINDOW.window_width", 1280);
        m_window_config.window_height      = pt.get<int>("WINDOW.window_height", 720);
        m_window_config.view_window_width  = pt.get<int>("WINDOW.view_panel_width", 800);
        m_window_config.view_window_height = pt.get<int>("WINDOW.view_panel_height", 720);
        m_window_config.menu_window_width  = pt.get<int>("WINDOW.menu_panel_width", 480);
        m_window_config.menu_window_height = pt.get<int>("WINDOW.menu_panel_height", 720);
        m_window_config.view_background_r  = pt.get<int>("COLORS.view_panel_background_r", 0);
        m_window_config.view_background_g  = pt.get<int>("COLORS.view_panel_background_g", 41);
        m_window_config.view_background_b  = pt.get<int>("COLORS.view_panel_background_b", 58);
        m_window_config.menu_background_r  = pt.get<int>("COLORS.menu_panel_background_r", 211);
        m_window_config.menu_background_g  = pt.get<int>("COLORS.menu_panel_background_g", 211);
        m_window_config.menu_background_b  = pt.get<int>("COLORS.menu_panel_background_b", 211);

        m_draw_config.draw_interval = pt.get<int>("DRAWING.drawing_interval_ms", 500);
        m_draw_config.max_tracks = pt.get<int>("DRAWING.max_tracks", 50);
        m_draw_config.draw_range = pt.get<double>("DRAWING.drawing_max_range_m", 60);
        m_draw_config.track_scale = pt.get<double>("DRAWING.track_scale", 1);
        m_draw_config.draw_step = pt.get<double>("DRAWING.drawing_step_m", 10);

        m_slider_config.label_size = pt.get<int>("SLIDER.label_size", 8);
        m_slider_config.label_spacing = pt.get<int>("SLIDER.label_spacing", 35);

        m_tracker_config.distance_threshold = pt.get<double>("TRACKER.distance_threshold_m", 100);
        m_tracker_config.bearing_threshold = pt.get<double>("TRACKER.bearing_threshold_deg", 15);
        m_tracker_config.heading_threshold = pt.get<double>("TRACKER.heading_threshold_deg", 10);
        m_tracker_config.speed_threshold = pt.get<double>("TRACKER.speed_threshold_kts", 7);
        m_tracker_config.track_timeout = pt.get<long long>("TRACKER.track_timeout_sec", 60) * 1000;

        m_connection_config.ip_address = pt.get<std::string>("CONNECTIONS.ip_address", "224.0.0.1");
        m_connection_config.port = pt.get<int>("CONNECTIONS.port", 5000);
        m_connection_config.interface = pt.get<std::string>("CONNECTIONS.interface", "0.0.0.0");

        BOOST_LOG_SEV(m_logger, info) << "    APP Name: " << m_window_config.app_name;
        BOOST_LOG_SEV(m_logger, info) << "    Window size: " << m_window_config.window_width << "x" << m_window_config.window_height;
        BOOST_LOG_SEV(m_logger, info) << "    View panel size: " << m_window_config.view_window_width << "x" << m_window_config.view_window_height;
        BOOST_LOG_SEV(m_logger, info) << "    Menu panel size: " << m_window_config.menu_window_width << "x" << m_window_config.menu_window_height;
        BOOST_LOG_SEV(m_logger, info) << "    View panel background color: rgb(" << m_window_config.view_background_r << ", "
                                      << m_window_config.view_background_g << ", " << m_window_config.view_background_b << ")";
        BOOST_LOG_SEV(m_logger, info) << "    Menu panel background color: rgb(" << m_window_config.menu_background_r << ", "
                                      << m_window_config.menu_background_g << ", " << m_window_config.menu_background_b << ")";

        BOOST_LOG_SEV(m_logger, info) << "    Drawing interval: " << m_draw_config.draw_interval;
        BOOST_LOG_SEV(m_logger, info) << "    Max tracks: " << m_draw_config.max_tracks;
        BOOST_LOG_SEV(m_logger, info) << "    Drawing range: " << m_draw_config.draw_range;
        BOOST_LOG_SEV(m_logger, info) << "    Track scale: " << m_draw_config.track_scale;
        BOOST_LOG_SEV(m_logger, info) << "    Drawing step: " << m_draw_config.draw_step;

        BOOST_LOG_SEV(m_logger, info) << "    Slider label size: " << m_slider_config.label_size;
        BOOST_LOG_SEV(m_logger, info) << "    Slider label spacing: " << m_slider_config.label_spacing;

        BOOST_LOG_SEV(m_logger, info) << "    Distance threshold (m): " << m_tracker_config.distance_threshold;
        BOOST_LOG_SEV(m_logger, info) << "    Bearing threshold (deg): " << m_tracker_config.bearing_threshold;
        BOOST_LOG_SEV(m_logger, info) << "    Heading threshold (deg): " << m_tracker_config.heading_threshold;
        BOOST_LOG_SEV(m_logger, info) << "    Speed threshold (kts): " << m_tracker_config.speed_threshold;
        BOOST_LOG_SEV(m_logger, info) << "    Track timeout (msec): " << m_tracker_config.track_timeout;

        BOOST_LOG_SEV(m_logger, info) << "    IP address: " << m_connection_config.ip_address;
        BOOST_LOG_SEV(m_logger, info) << "    Port: " << m_connection_config.port;
        BOOST_LOG_SEV(m_logger, info) << "    Interface: " << m_connection_config.interface;

    } catch (const boost::property_tree::ptree_error &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        BOOST_LOG_SEV(m_logger, fatal) << "    Configuration file couldnt be opened, aborting";
        return;
    }
}
