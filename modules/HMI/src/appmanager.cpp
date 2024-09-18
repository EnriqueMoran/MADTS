#include "appmanager.h"

namespace logging = boost::log;
namespace attrs = logging::attributes;
using namespace logging::trivial;

AppManager::AppManager()
{
    initializeLogger();

    BOOST_LOG_SEV(m_logger, info) << "Application started...";

    ConfigurationManager *config_manager = ConfigurationManager::getInstance();

    m_tracker = new Tracker();
    m_draw_manager = new DrawManager(m_tracker);
    m_multicast_manager = new MulticastManager(m_tracker);
}

AppManager::~AppManager()
{
    delete m_draw_manager;
    delete m_tracker;
}

void AppManager::run()
{
    m_multicast_manager->moveToThread(&m_multicast_manager->m_thread);
    m_multicast_manager->m_thread.start();
    BOOST_LOG_SEV(m_logger, info) << "Multicast thread started";
}

void AppManager::initializeLogger()
{
    logging::add_file_log
        (
            logging::keywords::file_name = "HMI.log",
            logging::keywords::rotation_size = 150 * 1024 * 1024,    // Rotate every 150 MiB
            logging::keywords::format = "[%TimeStamp%] - [%Severity%]: %Message%",
            logging::keywords::auto_flush = true
        );

    boost::shared_ptr< logging::core > core = logging::core::get();
    core->add_global_attribute("LineID", attrs::counter< unsigned int >(1));
    core->add_global_attribute("TimeStamp", attrs::local_clock());

    logging::core::get()->set_filter
        (
            logging::trivial::severity >= logging::trivial::trace
        );


}
