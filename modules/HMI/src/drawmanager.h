#ifndef DRAWMANAGER_H
#define DRAWMANAGER_H

#include "mainwindow.h"
#include "tracker.h"
#include "clickablelabel.h"

#include <QObject>
#include <QTimer>
#include <QLabel>
#include <QSlider>
#include <QCheckBox>
#include <QLineEdit>
#include <QPushButton>
#include <boost/log/trivial.hpp>

class DrawManager : public QObject
{
    Q_OBJECT

public:
    DrawManager(Tracker *tracker);
    ~DrawManager();

private:
    void initialize();
    void initializeTracks();
    void initializeMenu();
    void drawReference();
    void getTrackPosition(int &pos_x, int &pos_y, double distance, double bearing);
    void addHeadingBar(Track *track, QLabel *label, QPixmap &base_pixmap);

    QPixmap getTrackImage(Track *track);

    boost::log::sources::severity_logger<boost::log::trivial::severity_level> m_logger;

    int m_max_tracks;
    int m_drawing_interval;
    double m_max_range;       // Representation range
    double m_track_scale;
    double m_step_dist;    // Distance between steps
    bool m_show_track_info;

    MainWindow m_main_window;
    Tracker *m_tracker;
    QTimer *m_draw_timer;
    QTimer *m_update_track_info_timer;
    QMap<int, ClickableLabel*> m_label_map;
    QMap<int, QLabel*> m_info_label_map;

    QPixmap air_ally_img;
    QPixmap air_enemy_img;
    QPixmap air_unknown_img;
    QPixmap surface_ally_img;
    QPixmap surface_enemy_img;
    QPixmap surface_unknown_img;
    QPixmap gray_bar_long_img;
    QPixmap gray_bar_short_img;
    QPixmap blue_bar_mid_img;
    QPixmap yellow_bar_mid_img;
    QPixmap red_bar_mid_img;
    QPixmap ownship_mark_img;
    QPixmap surface_selector_img;
    QPixmap air_selector_img;

    QLabel *m_ownship_label;
    QLabel *m_horizontal_axis;
    QLabel *m_vertical_axis;

    QLabel *m_horizontal_measure_1;
    QLabel *m_horizontal_measure_2;
    QLabel *m_horizontal_measure_3;
    QLabel *m_horizontal_measure_4;
    QLabel *m_horizontal_measure_5;
    QLabel *m_horizontal_measure_6;

    QLabel *m_vertical_measure_1;
    QLabel *m_vertical_measure_2;
    QLabel *m_vertical_measure_3;
    QLabel *m_vertical_measure_4;
    QLabel *m_vertical_measure_5;
    QLabel *m_vertical_measure_6;

    QLabel *m_horizontal_measure_1_text;
    QLabel *m_horizontal_measure_2_text;
    QLabel *m_horizontal_measure_3_text;
    QLabel *m_horizontal_measure_4_text;
    QLabel *m_horizontal_measure_5_text;
    QLabel *m_horizontal_measure_6_text;

    QLabel *m_vertical_measure_1_text;
    QLabel *m_vertical_measure_2_text;
    QLabel *m_vertical_measure_3_text;
    QLabel *m_vertical_measure_4_text;
    QLabel *m_vertical_measure_5_text;
    QLabel *m_vertical_measure_6_text;

    QSlider *m_range_slider;
    QLabel *m_slider_text;
    QWidget *m_slider_values;

    QLabel *m_checkbox_text;
    QCheckBox *m_show_info_checkbox;

    QLabel *m_track_info_title;
    QLabel *m_track_distance;
    QLabel *m_track_bearing;
    QLabel *m_track_speed;
    QLabel *m_track_heading;
    QLabel *m_track_type;
    QLabel *m_track_behavior;

    QLineEdit *m_track_distance_value;
    QLineEdit *m_track_bearing_value;
    QLineEdit *m_track_speed_value;
    QLineEdit *m_track_heading_value;
    QLineEdit *m_track_type_value;
    QLineEdit *m_track_behavior_value;

    QPushButton *m_deselec_button;

    int m_selected_track_id;    // -1 means no track is selected

private slots:
    void drawTracks();
    void sliderValueChanged(int new_value);
    void checkboxValueChanged(int new_state);
    void trackSelected(Track *track);
    void unselecTrack();
    void updateSelectedTrackInfo();
};

#endif // DRAWMANAGER_H
