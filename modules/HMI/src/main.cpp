
#include "appmanager.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication::setDesktopSettingsAware(false);

    QApplication a(argc, argv);

    AppManager app_manager;
    app_manager.run();

    return a.exec();
}
