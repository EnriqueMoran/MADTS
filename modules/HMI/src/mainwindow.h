#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QGridLayout>

#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/sources/severity_logger.hpp>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class DrawManager;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    friend class DrawManager;

private:
    // Configure window's properties
    void initialize();

    Ui::MainWindow *ui;
    std::unique_ptr<QWidget> m_central_widget;
    std::unique_ptr<QGridLayout> m_layout_ptr;
    std::unique_ptr<QWidget> m_menu_widget_ptr;
    std::unique_ptr<QWidget> m_view_widget_ptr;

    boost::log::sources::severity_logger<boost::log::trivial::severity_level> m_logger;
};
#endif // MAINWINDOW_H
