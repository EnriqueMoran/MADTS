#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "configurationmanager.h"

#include <QScreen>
#include <QGuiApplication>

namespace logging = boost::log;
namespace attrs = logging::attributes;
using namespace logging::trivial;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    initialize();
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::initialize()
{
    // Set APP Title
    setWindowTitle(QString::fromStdString(ConfigurationManager::getInstance()->getWindowConfig().app_name));

    // Set Main window fixed size
    int main_window_width = ConfigurationManager::getInstance()->getWindowConfig().window_width;
    int main_window_height = ConfigurationManager::getInstance()->getWindowConfig().window_height;
    this->setFixedSize(main_window_width, main_window_height);
    BOOST_LOG_SEV(m_logger, info) << "Setting main window size: " << main_window_width << "x" << main_window_height;

    QScreen *screen = QGuiApplication::primaryScreen();
    QRect screenGeometry = screen->geometry();
    int x = (screenGeometry.width() - width()) / 2;
    int y = (screenGeometry.height() - height()) / 2;
    move(x, y);


    // Create main view and side menu widgets
    m_central_widget = std::make_unique<QWidget>(this);
    setCentralWidget(m_central_widget.get());

    m_layout_ptr = std::make_unique<QGridLayout>(this);
    m_layout_ptr->setContentsMargins(0, 0, 0, 0);
    m_layout_ptr->setSpacing(0);

    m_menu_widget_ptr = std::make_unique<QWidget>();
    m_view_widget_ptr = std::make_unique<QWidget>();

    // Set widgets background color
    int view_r = ConfigurationManager::getInstance()->getWindowConfig().view_background_r;
    int view_g = ConfigurationManager::getInstance()->getWindowConfig().view_background_g;
    int view_b = ConfigurationManager::getInstance()->getWindowConfig().view_background_b;

    std::string view_background = "rgb(" + std::to_string(view_r) + ", " + std::to_string(view_g) + ", " + std::to_string(view_b) + ");";
    m_view_widget_ptr->setStyleSheet("background-color: " + QString::fromStdString(view_background));

    int menu_r = ConfigurationManager::getInstance()->getWindowConfig().menu_background_r;
    int menu_g = ConfigurationManager::getInstance()->getWindowConfig().menu_background_g;
    int menu_b = ConfigurationManager::getInstance()->getWindowConfig().menu_background_b;
    std::string menu_background = "rgb(" + std::to_string(menu_r) + ", " + std::to_string(menu_g) + ", " + std::to_string(menu_b) + ");";
    m_menu_widget_ptr->setStyleSheet("background-color: " + QString::fromStdString(menu_background));

    // Set widget size
    int view_width = ConfigurationManager::getInstance()->getWindowConfig().view_window_width;
    int view_height = ConfigurationManager::getInstance()->getWindowConfig().view_window_height;
    m_view_widget_ptr->setFixedSize(view_width, view_height);

    int panel_width = ConfigurationManager::getInstance()->getWindowConfig().menu_window_width;
    int panel_height = ConfigurationManager::getInstance()->getWindowConfig().menu_window_height;
    m_menu_widget_ptr->setFixedSize(panel_width, panel_height);

    m_layout_ptr->addWidget(m_view_widget_ptr.get(), 0, 0);
    m_layout_ptr->addWidget(m_menu_widget_ptr.get(), 0, 1);

    m_central_widget->setLayout(m_layout_ptr.get());
}
