TEMPLATE = app

ROOTDIR = $$_PRO_FILE_PWD_

QT       += core gui network

QMAKE_CXXFLAGS += "-fno-sized-deallocation"

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

include(lib/common/paths.pri)

cfg.files = $$ROOTDIR/cfg/*
cfg.path = $$CFG_DEST_DIR/
message("Adding cfg files:" $$cfg.path)
INSTALLS += cfg

src.files = $$ROOTDIR/src/*
src.path = $$SRC_DEST_DIR/
message("Adding src files:" $$src.path)
INSTALLS += src

SOURCES += \
    src/clickablelabel.cpp \
    src/appmanager.cpp \
    src/configurationmanager.cpp \
    src/drawmanager.cpp \
    src/main.cpp \
    src/mainwindow.cpp \
    src/multicastmanager.cpp \
    src/track.cpp \
    src/tracker.cpp \
    src/trackmessage.cpp

HEADERS += \
    src/clickablelabel.h \
    src/appmanager.h \
    src/configurationmanager.h \
    src/drawmanager.h \
    src/mainwindow.h \
    src/multicastmanager.h \
    src/track.h \
    src/tracker.h \
    src/trackmessage.h

FORMS += \
    src/mainwindow.ui

DEFINES += BOOST_LOG_DYN_LINK

win32: {
    LIBS += -LC:/Users/Enrik/Documents/01-Programas/boost_1_86_0/stage/lib
    INCLUDEPATH += C:/Users/Enrik/Documents/01-Programas/boost_1_86_0
    LIBS += -lws2_32
    LIBS += -lboost_system-mgw11-mt-x64-1_86
    LIBS += -lboost_filesystem-mgw11-mt-x64-1_86
    LIBS += -lboost_thread-mgw11-mt-x64-1_86
    LIBS += -lboost_log-mgw11-mt-x64-1_86
    LIBS += -lboost_log_setup-mgw11-mt-x64-1_86
}

unix:!macx: {
    LIBS += -L$$system(echo ${BOOST_LIB_DIR})
    LIBS += -L$$PWD/../../../../../../usr/lib/x86_64-linux-gnu/ -lboost_system
    INCLUDEPATH += $$PWD/../../../../../../usr/lib/x86_64-linux-gnu
    DEPENDPATH += $$PWD/../../../../../../usr/lib/x86_64-linux-gnu
}

target.path = $$DEST_DIR/
message("Adding bin files:" $$target.path)

RESOURCES += \
    src/resource.qrc
