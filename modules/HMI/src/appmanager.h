#ifndef APPMANAGER_H
#define APPMANAGER_H

#include "drawmanager.h"
#include "tracker.h"
#include "configurationmanager.h"
#include "multicastmanager.h"

#include <fstream>
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/sources/severity_logger.hpp>

class AppManager
{
public:
    AppManager();
    ~AppManager();

    void run();

private:
    void initializeLogger();
    Tracker *m_tracker;
    DrawManager *m_draw_manager;
    MulticastManager *m_multicast_manager;

    boost::log::sources::severity_logger<boost::log::trivial::severity_level> m_logger;
};

#endif // APPMANAGER_H
