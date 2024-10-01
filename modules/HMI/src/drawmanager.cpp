#include "drawmanager.h"
#include "configurationmanager.h"

#include <QLabel>
#include <QPalette>
#include <QPainter>
#include <QFontMetrics>
#include <iostream>
#include <cmath>


using namespace boost::log::trivial;

DrawManager::DrawManager(Tracker *tracker)
{
    m_tracker = tracker;

    m_drawing_interval = ConfigurationManager::getInstance()->getDrawConfig().draw_interval;
    m_max_tracks = ConfigurationManager::getInstance()->getDrawConfig().max_tracks;
    m_max_range = ConfigurationManager::getInstance()->getDrawConfig().draw_range;
    m_step_dist = ConfigurationManager::getInstance()->getDrawConfig().draw_step;
    m_track_scale = ConfigurationManager::getInstance()->getDrawConfig().track_scale;
    m_draw_timer = new QTimer(this);
    m_update_track_info_timer = new QTimer(this);
    m_show_track_info = false;
    m_selected_track_id = -1;
    initialize();
}

DrawManager::~DrawManager()
{
    for (auto it = m_label_map.begin(); it != m_label_map.end(); ++it) {
        delete it.value();
    }

    for (auto it = m_info_label_map.begin(); it != m_info_label_map.end(); ++it) {
        delete it.value();
    }

    m_info_label_map.clear();

    m_label_map.clear();
}

void DrawManager::initialize()
{
    connect(m_draw_timer, SIGNAL(timeout()), this, SLOT(drawTracks()));
    m_draw_timer->start(m_drawing_interval);

    connect(m_update_track_info_timer, SIGNAL(timeout()), this, SLOT(updateSelectedTrackInfo()));
    m_update_track_info_timer->start(m_drawing_interval);

    air_ally_img.load(":/img/resources/img/air_ally.png");
    air_enemy_img.load(":/img/resources/img/air_enemy.png");
    air_unknown_img.load(":/img/resources/img/air_unknown.png");
    surface_ally_img.load(":/img/resources/img/surface_ally.png");
    surface_enemy_img.load(":/img/resources/img/surface_enemy.png");
    surface_unknown_img.load(":/img/resources/img/surface_unknown.png");
    gray_bar_long_img.load(":/img/resources/img/gray_bar_long.png");
    gray_bar_short_img.load(":/img/resources/img/gray_bar_short.png");
    blue_bar_mid_img.load(":/img/resources/img/blue_bar_mid.png");
    red_bar_mid_img.load(":/img/resources/img/red_bar_mid.png");
    yellow_bar_mid_img.load(":/img/resources/img/yellow_bar_mid.png");
    ownship_mark_img.load(":/img/resources/img/ownship_bars.png");
    surface_selector_img.load(":/img/resources/img/surface_selector.png");
    air_selector_img.load(":/img/resources/img/air_selector.png");

    drawReference();
    initializeMenu();
    initializeTracks();
    m_main_window.show();
}

void DrawManager::initializeTracks()
{
    m_label_map.clear();
    for (int i=0; i < m_max_tracks; i++)
    {
        ClickableLabel *label = new ClickableLabel(m_main_window.m_view_widget_ptr.get());
        label->setScaledContents(false);
        label->setAlignment(Qt::AlignCenter);
        label->setVisible(false);
        m_label_map.insert(i, label);

        QLabel *infolabel = new QLabel(m_main_window.m_view_widget_ptr.get());
        infolabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        m_info_label_map.insert(i, infolabel);
    }
}

void DrawManager::initializeMenu()
{
    m_range_slider = new QSlider(Qt::Horizontal);
    m_range_slider->setMinimum((m_max_range / m_step_dist) - 5);    // 5 is the number of steps we want in the slider
    m_range_slider->setMaximum(m_max_range / m_step_dist);
    m_range_slider->setSingleStep(1);
    m_range_slider->setTickInterval(1);
    m_range_slider->setTickPosition(QSlider::TicksBelow);
    m_range_slider->setValue(m_max_range / m_step_dist);

    connect(m_range_slider, &QSlider::valueChanged, this, &DrawManager::sliderValueChanged);


    QString sliderStyle = "QSlider::groove:horizontal {"
                          "    border: 1px solid #999999;"
                          "    height: 4px;"
                          "    margin: 0px;"
                          "    background: transparent;"
                          "}"

                          "QSlider::handle:horizontal {"
                          "    background: #FFFFFF;"
                          "    border: 1px solid #555555;"
                          "    width: 10px;"
                          "    margin: -5px 0px;"
                          "    border-radius: 2px;"
                          "}";

    m_range_slider->setStyleSheet(sliderStyle);

    m_slider_text = new QLabel("View range (m):");

    QGridLayout *menuLayout = new QGridLayout(m_main_window.m_menu_widget_ptr.get());
    menuLayout->setContentsMargins(10, 40, 20, 0);

    menuLayout->addWidget(m_slider_text, 0, 0, Qt::AlignTop);
    menuLayout->addWidget(m_range_slider, 0, 1, Qt::AlignTop);

    QHBoxLayout *tickLabelsLayout = new QHBoxLayout();

    for (int i = m_range_slider->minimum(); i <= m_range_slider->maximum(); i += m_range_slider->singleStep()) {
        QLabel *tickLabel = new QLabel(QString::number(i * m_step_dist, 'f', 2));
        tickLabel->setAlignment(Qt::AlignCenter);

        QFont font = tickLabel->font();
        font.setPointSize(8);
        tickLabel->setFont(font);

        tickLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
        tickLabel->setWordWrap(true);
        tickLabel->adjustSize();

        tickLabelsLayout->addWidget(tickLabel);
        tickLabelsLayout->setSpacing(35);
    }

    menuLayout->addLayout(tickLabelsLayout, 1, 1, 1, 2, Qt::AlignTop);
    m_main_window.m_menu_widget_ptr->setLayout(menuLayout);

    menuLayout->setRowStretch(2, 1);

    m_checkbox_text = new QLabel("Show tracks info");
    menuLayout->addWidget(m_checkbox_text, 3, 0, Qt::AlignTop);

    m_show_info_checkbox = new QCheckBox();
    m_show_info_checkbox->setChecked(false);
    menuLayout->addWidget(m_show_info_checkbox, 3, 1, Qt::AlignTop);

    connect(m_show_info_checkbox, &QCheckBox::stateChanged, this, &DrawManager::checkboxValueChanged);


    menuLayout->setRowStretch(4, 2);

    m_track_info_title = new QLabel("Selected Track Info");
    QFont title_font = m_track_info_title->font();
    title_font.setPointSize(14);
    title_font.setBold(true);
    m_track_info_title->setFont(title_font);
    menuLayout->addWidget(m_track_info_title, 5, 1, Qt::AlignTop);

    menuLayout->setRowStretch(6, 1);

    m_track_distance = new QLabel("Distance (m): ");

    QFont label_font = m_track_distance->font();
    label_font.setBold(true);
    m_track_distance->setFont(label_font);

    menuLayout->addWidget(m_track_distance, 7, 0, Qt::AlignTop);
    m_track_distance_value = new QLineEdit("No track selected");
    m_track_distance_value->setReadOnly(true);

    QString styleSheet = "selection-background-color: transparent;";
    m_track_distance_value->setStyleSheet(styleSheet);

    QPalette palette = m_track_distance_value->palette();
    palette.setColor(QPalette::Text, Qt::gray);

    m_track_distance_value->setPalette(palette);
    menuLayout->addWidget(m_track_distance_value, 7, 1, Qt::AlignTop);

    m_track_bearing = new QLabel("Bearing (DEG): ");
    m_track_bearing->setFont(label_font);
    menuLayout->addWidget(m_track_bearing, 8, 0, Qt::AlignTop);
    m_track_bearing_value = new QLineEdit("No track selected");
    m_track_bearing_value->setReadOnly(true);
    m_track_bearing_value->setStyleSheet(styleSheet);
    m_track_bearing_value->setPalette(palette);
    menuLayout->addWidget(m_track_bearing_value, 8, 1, Qt::AlignTop);

    m_track_heading = new QLabel("Heading (DEG): ");
    m_track_heading->setFont(label_font);
    menuLayout->addWidget(m_track_heading, 9, 0, Qt::AlignTop);
    m_track_heading_value = new QLineEdit("No track selected");
    m_track_heading_value->setReadOnly(true);
    m_track_heading_value->setStyleSheet(styleSheet);
    m_track_heading_value->setPalette(palette);
    menuLayout->addWidget(m_track_heading_value, 9, 1, Qt::AlignTop);

    m_track_speed = new QLabel("Speed (KTS): ");
    m_track_speed->setFont(label_font);
    menuLayout->addWidget(m_track_speed, 10, 0, Qt::AlignTop);
    m_track_speed_value = new QLineEdit("No track selected");
    m_track_speed_value->setReadOnly(true);
    m_track_speed_value->setStyleSheet(styleSheet);
    m_track_speed_value->setPalette(palette);
    menuLayout->addWidget(m_track_speed_value, 10, 1, Qt::AlignTop);

    m_track_type = new QLabel("Domain: ");
    m_track_type->setFont(label_font);
    menuLayout->addWidget(m_track_type, 11, 0, Qt::AlignTop);
    m_track_type_value = new QLineEdit("No track selected");
    m_track_type_value->setReadOnly(true);
    m_track_type_value->setStyleSheet(styleSheet);
    m_track_type_value->setPalette(palette);
    menuLayout->addWidget(m_track_type_value, 11, 1, Qt::AlignTop);

    m_track_behavior = new QLabel("Classification: ");
    m_track_behavior->setFont(label_font);
    menuLayout->addWidget(m_track_behavior, 12, 0, Qt::AlignTop);
    m_track_behavior_value = new QLineEdit("No track selected");
    m_track_behavior_value->setReadOnly(true);
    m_track_behavior_value->setStyleSheet(styleSheet);
    m_track_behavior_value->setPalette(palette);
    menuLayout->addWidget(m_track_behavior_value, 12, 1, Qt::AlignTop);

    menuLayout->setRowStretch(13, 1);

    m_deselec_button = new QPushButton("Clear selected track");
    m_deselec_button->show();
    menuLayout->addWidget(m_deselec_button, 14, 1, Qt::AlignTop);

    connect(m_deselec_button, &QPushButton::clicked, this, &DrawManager::unselecTrack);

    menuLayout->setRowStretch(15, 7);

}

void DrawManager::drawReference()
{
    int max_width = m_main_window.m_view_widget_ptr.get()->width();
    int max_height = m_main_window.m_view_widget_ptr.get()->height();

    QPixmap rotated_gray_bar_long_scaled = gray_bar_long_img.scaled(max_width, 1, Qt::IgnoreAspectRatio);
    QPixmap rotated_gray_bar_long = rotated_gray_bar_long_scaled.transformed(QTransform().rotate(90));

    QPixmap rotated_gray_bar_short_scaled = gray_bar_short_img.scaled(20, 1, Qt::IgnoreAspectRatio);
    QPixmap rotated_gray_bar_short = rotated_gray_bar_short_scaled.transformed(QTransform().rotate(90));

    m_horizontal_axis = new QLabel(m_main_window.m_view_widget_ptr.get());
    m_horizontal_axis->setPixmap(rotated_gray_bar_long_scaled);
    m_horizontal_axis->setAttribute(Qt::WA_TransparentForMouseEvents);
    m_horizontal_axis->setAlignment(Qt::AlignCenter);
    m_horizontal_axis->setVisible(true);
    m_horizontal_axis->setStyleSheet("background-color: transparent;");
    m_horizontal_axis->move(0, max_height / 2);

    m_vertical_axis = new QLabel(m_main_window.m_view_widget_ptr.get());
    m_vertical_axis->setPixmap(rotated_gray_bar_long);
    m_vertical_axis->setAttribute(Qt::WA_TransparentForMouseEvents);
    m_vertical_axis->setAlignment(Qt::AlignCenter);
    m_vertical_axis->setVisible(true);
    m_vertical_axis->setStyleSheet("background-color: transparent;");
    m_vertical_axis->move(max_width / 2, 0);

    m_ownship_label = new QLabel(m_main_window.m_view_widget_ptr.get());
    m_ownship_label->setPixmap(ownship_mark_img);
    m_ownship_label->setAttribute(Qt::WA_TransparentForMouseEvents);
    m_ownship_label->setAlignment(Qt::AlignCenter);
    m_ownship_label->setVisible(true);
    m_ownship_label->setStyleSheet("background-color: transparent;");
    m_ownship_label->move((max_width / 2) - (ownship_mark_img.width() / 2) + 1,
                          (max_height / 2) - (ownship_mark_img.height()) + 1);  //TODO: Adjust

    m_horizontal_measure_1 = new QLabel(m_main_window.m_view_widget_ptr.get());
    m_horizontal_measure_1->setPixmap(rotated_gray_bar_short);
    m_horizontal_measure_1->setAttribute(Qt::WA_TransparentForMouseEvents);
    m_horizontal_measure_1->setAlignment(Qt::AlignCenter);
    m_horizontal_measure_1->setVisible(true);
    m_horizontal_measure_1->setStyleSheet("background-color: transparent;");
    m_horizontal_measure_1->move((max_width / 2) + (max_width / 8), (max_height / 2) - rotated_gray_bar_short.height() / 2);
    m_horizontal_measure_1->setWordWrap(true);

    m_horizontal_measure_2 = new QLabel(m_main_window.m_view_widget_ptr.get());
    m_horizontal_measure_2->setPixmap(rotated_gray_bar_short);
    m_horizontal_measure_2->setAttribute(Qt::WA_TransparentForMouseEvents);
    m_horizontal_measure_2->setAlignment(Qt::AlignCenter);
    m_horizontal_measure_2->setVisible(true);
    m_horizontal_measure_2->setStyleSheet("background-color: transparent;");
    m_horizontal_measure_2->move((max_width / 2) + (max_width / 8)*2, (max_height / 2) - rotated_gray_bar_short.height() / 2);
    m_horizontal_measure_2->setWordWrap(true);

    m_horizontal_measure_3 = new QLabel(m_main_window.m_view_widget_ptr.get());
    m_horizontal_measure_3->setPixmap(rotated_gray_bar_short);
    m_horizontal_measure_3->setAttribute(Qt::WA_TransparentForMouseEvents);
    m_horizontal_measure_3->setAlignment(Qt::AlignCenter);
    m_horizontal_measure_3->setVisible(true);
    m_horizontal_measure_3->setStyleSheet("background-color: transparent;");
    m_horizontal_measure_3->move((max_width / 2) + (max_width / 8)*3, (max_height / 2) - rotated_gray_bar_short.height() / 2);
    m_horizontal_measure_3->setWordWrap(true);

    m_horizontal_measure_4 = new QLabel(m_main_window.m_view_widget_ptr.get());
    m_horizontal_measure_4->setPixmap(rotated_gray_bar_short);
    m_horizontal_measure_4->setAttribute(Qt::WA_TransparentForMouseEvents);
    m_horizontal_measure_4->setAlignment(Qt::AlignCenter);
    m_horizontal_measure_4->setVisible(true);
    m_horizontal_measure_4->setStyleSheet("background-color: transparent;");
    m_horizontal_measure_4->move((max_width / 2) - (max_width / 8), (max_height / 2) - rotated_gray_bar_short.height() / 2);
    m_horizontal_measure_4->setWordWrap(true);

    m_horizontal_measure_5 = new QLabel(m_main_window.m_view_widget_ptr.get());
    m_horizontal_measure_5->setPixmap(rotated_gray_bar_short);
    m_horizontal_measure_5->setAttribute(Qt::WA_TransparentForMouseEvents);
    m_horizontal_measure_5->setAlignment(Qt::AlignCenter);
    m_horizontal_measure_5->setVisible(true);
    m_horizontal_measure_5->setStyleSheet("background-color: transparent;");
    m_horizontal_measure_5->move((max_width / 2) - (max_width / 8)*2, (max_height / 2) - rotated_gray_bar_short.height() / 2);
    m_horizontal_measure_5->setWordWrap(true);

    m_horizontal_measure_6 = new QLabel(m_main_window.m_view_widget_ptr.get());
    m_horizontal_measure_6->setPixmap(rotated_gray_bar_short);
    m_horizontal_measure_6->setAttribute(Qt::WA_TransparentForMouseEvents);
    m_horizontal_measure_6->setAlignment(Qt::AlignCenter);
    m_horizontal_measure_6->setVisible(true);
    m_horizontal_measure_6->setStyleSheet("background-color: transparent;");
    m_horizontal_measure_6->move((max_width / 2) - (max_width / 8)*3, (max_height / 2) - rotated_gray_bar_short.height() / 2);
    m_horizontal_measure_6->setWordWrap(true);

    m_vertical_measure_1 = new QLabel(m_main_window.m_view_widget_ptr.get());
    m_vertical_measure_1->setPixmap(rotated_gray_bar_short_scaled);
    m_vertical_measure_1->setAttribute(Qt::WA_TransparentForMouseEvents);
    m_vertical_measure_1->setAlignment(Qt::AlignCenter);
    m_vertical_measure_1->setVisible(true);
    m_vertical_measure_1->setStyleSheet("background-color: transparent;");
    m_vertical_measure_1->move((max_width / 2) - rotated_gray_bar_short_scaled.width() / 2, (max_height / 2) + (max_height / 8));
    m_vertical_measure_1->setWordWrap(true);

    m_vertical_measure_2 = new QLabel(m_main_window.m_view_widget_ptr.get());
    m_vertical_measure_2->setPixmap(rotated_gray_bar_short_scaled);
    m_vertical_measure_2->setAttribute(Qt::WA_TransparentForMouseEvents);
    m_vertical_measure_2->setAlignment(Qt::AlignCenter);
    m_vertical_measure_2->setVisible(true);
    m_vertical_measure_2->setStyleSheet("background-color: transparent;");
    m_vertical_measure_2->move((max_width / 2) - rotated_gray_bar_short_scaled.width() / 2, (max_height / 2) + (max_height / 8)*2);
    m_vertical_measure_2->setWordWrap(true);

    m_vertical_measure_3 = new QLabel(m_main_window.m_view_widget_ptr.get());
    m_vertical_measure_3->setPixmap(rotated_gray_bar_short_scaled);
    m_vertical_measure_3->setAttribute(Qt::WA_TransparentForMouseEvents);
    m_vertical_measure_3->setAlignment(Qt::AlignCenter);
    m_vertical_measure_3->setVisible(true);
    m_vertical_measure_3->setStyleSheet("background-color: transparent;");
    m_vertical_measure_3->move((max_width / 2) - rotated_gray_bar_short_scaled.width() / 2, (max_height / 2) + (max_height / 8)*3);
    m_vertical_measure_3->setWordWrap(true);

    m_vertical_measure_4 = new QLabel(m_main_window.m_view_widget_ptr.get());
    m_vertical_measure_4->setPixmap(rotated_gray_bar_short_scaled);
    m_vertical_measure_4->setAttribute(Qt::WA_TransparentForMouseEvents);
    m_vertical_measure_4->setAlignment(Qt::AlignCenter);
    m_vertical_measure_4->setVisible(true);
    m_vertical_measure_4->setStyleSheet("background-color: transparent;");
    m_vertical_measure_4->move((max_width / 2) - rotated_gray_bar_short_scaled.width() / 2, (max_height / 2) - (max_height / 8));
    m_vertical_measure_4->setWordWrap(true);

    m_vertical_measure_5 = new QLabel(m_main_window.m_view_widget_ptr.get());
    m_vertical_measure_5->setPixmap(rotated_gray_bar_short_scaled);
    m_vertical_measure_5->setAttribute(Qt::WA_TransparentForMouseEvents);
    m_vertical_measure_5->setAlignment(Qt::AlignCenter);
    m_vertical_measure_5->setVisible(true);
    m_vertical_measure_5->setStyleSheet("background-color: transparent;");
    m_vertical_measure_5->move((max_width / 2) - rotated_gray_bar_short_scaled.width() / 2, (max_height / 2) - (max_height / 8)*2);
    m_vertical_measure_5->setWordWrap(true);

    m_vertical_measure_6 = new QLabel(m_main_window.m_view_widget_ptr.get());
    m_vertical_measure_6->setPixmap(rotated_gray_bar_short_scaled);
    m_vertical_measure_6->setAttribute(Qt::WA_TransparentForMouseEvents);
    m_vertical_measure_6->setAlignment(Qt::AlignCenter);
    m_vertical_measure_6->setVisible(true);
    m_vertical_measure_6->setStyleSheet("background-color: transparent;");
    m_vertical_measure_6->move((max_width / 2) - rotated_gray_bar_short_scaled.width() / 2, (max_height / 2) - (max_height / 8)*3);
    m_vertical_measure_6->setWordWrap(true);

    double h_measure_1_value = (m_max_range / 4.) * 1;
    double h_measure_2_value = (m_max_range / 4.) * 2;
    double h_measure_3_value = (m_max_range / 4.) * 3;
    double h_measure_4_value = (m_max_range / 4.) * 1;
    double h_measure_5_value = (m_max_range / 4.) * 2;
    double h_measure_6_value = (m_max_range / 4.) * 3;

    double v_measure_1_value = (m_max_range / 4.) * 1;
    double v_measure_2_value = (m_max_range / 4.) * 2;
    double v_measure_3_value = (m_max_range / 4.) * 3;
    double v_measure_4_value = (m_max_range / 4.) * 1;
    double v_measure_5_value = (m_max_range / 4.) * 2;
    double v_measure_6_value = (m_max_range / 4.) * 3;

    QString h_measure_1_value_text = std::fmod(h_measure_1_value, 1.0) == 0.0 ? QString::number(h_measure_1_value, 'f', 0) : QString::number(h_measure_1_value, 'f', 2);
    QString h_measure_2_value_text = std::fmod(h_measure_2_value, 1.0) == 0.0 ? QString::number(h_measure_2_value, 'f', 0) : QString::number(h_measure_2_value, 'f', 2);
    QString h_measure_3_value_text = std::fmod(h_measure_3_value, 1.0) == 0.0 ? QString::number(h_measure_3_value, 'f', 0) : QString::number(h_measure_3_value, 'f', 2);
    QString h_measure_4_value_text = std::fmod(h_measure_4_value, 1.0) == 0.0 ? QString::number(h_measure_4_value, 'f', 0) : QString::number(h_measure_4_value, 'f', 2);
    QString h_measure_5_value_text = std::fmod(h_measure_5_value, 1.0) == 0.0 ? QString::number(h_measure_5_value, 'f', 0) : QString::number(h_measure_5_value, 'f', 2);
    QString h_measure_6_value_text = std::fmod(h_measure_6_value, 1.0) == 0.0 ? QString::number(h_measure_6_value, 'f', 0) : QString::number(h_measure_6_value, 'f', 2);

    QString v_measure_1_value_text = std::fmod(v_measure_1_value, 1.0) == 0.0 ? QString::number(v_measure_1_value, 'f', 0) : QString::number(v_measure_1_value, 'f', 2);
    QString v_measure_2_value_text = std::fmod(v_measure_2_value, 1.0) == 0.0 ? QString::number(v_measure_2_value, 'f', 0) : QString::number(v_measure_2_value, 'f', 2);
    QString v_measure_3_value_text = std::fmod(v_measure_3_value, 1.0) == 0.0 ? QString::number(v_measure_3_value, 'f', 0) : QString::number(v_measure_3_value, 'f', 2);
    QString v_measure_4_value_text = std::fmod(v_measure_4_value, 1.0) == 0.0 ? QString::number(v_measure_4_value, 'f', 0) : QString::number(v_measure_4_value, 'f', 2);
    QString v_measure_5_value_text = std::fmod(v_measure_5_value, 1.0) == 0.0 ? QString::number(v_measure_5_value, 'f', 0) : QString::number(v_measure_5_value, 'f', 2);
    QString v_measure_6_value_text = std::fmod(v_measure_6_value, 1.0) == 0.0 ? QString::number(v_measure_6_value, 'f', 0) : QString::number(v_measure_6_value, 'f', 2);

    m_horizontal_measure_1_text = new QLabel(h_measure_1_value_text, m_main_window.m_view_widget_ptr.get());
    m_horizontal_measure_2_text = new QLabel(h_measure_2_value_text, m_main_window.m_view_widget_ptr.get());
    m_horizontal_measure_3_text = new QLabel(h_measure_3_value_text + " m", m_main_window.m_view_widget_ptr.get());
    m_horizontal_measure_4_text = new QLabel(h_measure_4_value_text, m_main_window.m_view_widget_ptr.get());
    m_horizontal_measure_5_text = new QLabel(h_measure_5_value_text, m_main_window.m_view_widget_ptr.get());
    m_horizontal_measure_6_text = new QLabel(h_measure_6_value_text, m_main_window.m_view_widget_ptr.get());

    m_vertical_measure_1_text = new QLabel(v_measure_1_value_text, m_main_window.m_view_widget_ptr.get());
    m_vertical_measure_2_text = new QLabel(v_measure_2_value_text, m_main_window.m_view_widget_ptr.get());
    m_vertical_measure_3_text = new QLabel(v_measure_3_value_text, m_main_window.m_view_widget_ptr.get());
    m_vertical_measure_4_text = new QLabel(v_measure_4_value_text, m_main_window.m_view_widget_ptr.get());
    m_vertical_measure_5_text = new QLabel(v_measure_5_value_text, m_main_window.m_view_widget_ptr.get());
    m_vertical_measure_6_text = new QLabel(v_measure_6_value_text + " m", m_main_window.m_view_widget_ptr.get());

    m_horizontal_measure_1_text->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    m_horizontal_measure_1_text->setWordWrap(true);
    m_horizontal_measure_2_text->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    m_horizontal_measure_2_text->setWordWrap(true);
    m_horizontal_measure_3_text->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    m_horizontal_measure_3_text->setWordWrap(true);
    m_horizontal_measure_4_text->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    m_horizontal_measure_4_text->setWordWrap(true);
    m_horizontal_measure_5_text->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    m_horizontal_measure_5_text->setWordWrap(true);
    m_horizontal_measure_6_text->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    m_horizontal_measure_6_text->setWordWrap(true);

    m_vertical_measure_1_text->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    m_vertical_measure_1_text->setWordWrap(true);
    m_vertical_measure_2_text->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    m_vertical_measure_2_text->setWordWrap(true);
    m_vertical_measure_3_text->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    m_vertical_measure_3_text->setWordWrap(true);
    m_vertical_measure_4_text->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    m_vertical_measure_4_text->setWordWrap(true);
    m_vertical_measure_5_text->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    m_vertical_measure_5_text->setWordWrap(true);
    m_vertical_measure_6_text->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    m_vertical_measure_6_text->setWordWrap(true);

    QPalette palette;
    QColor textColor(230, 230, 230);
    palette.setBrush(QPalette::WindowText, textColor);

    m_horizontal_measure_1_text->setPalette(palette);
    m_horizontal_measure_1_text->setStyleSheet("background-color: transparent;");
    m_horizontal_measure_2_text->setPalette(palette);
    m_horizontal_measure_2_text->setStyleSheet("background-color: transparent;");
    m_horizontal_measure_3_text->setPalette(palette);
    m_horizontal_measure_3_text->setStyleSheet("background-color: transparent;");
    m_horizontal_measure_4_text->setPalette(palette);
    m_horizontal_measure_4_text->setStyleSheet("background-color: transparent;");
    m_horizontal_measure_5_text->setPalette(palette);
    m_horizontal_measure_5_text->setStyleSheet("background-color: transparent;");
    m_horizontal_measure_6_text->setPalette(palette);
    m_horizontal_measure_6_text->setStyleSheet("background-color: transparent;");

    m_vertical_measure_1_text->setPalette(palette);
    m_vertical_measure_1_text->setStyleSheet("background-color: transparent;");
    m_vertical_measure_2_text->setPalette(palette);
    m_vertical_measure_2_text->setStyleSheet("background-color: transparent;");
    m_vertical_measure_3_text->setPalette(palette);
    m_vertical_measure_3_text->setStyleSheet("background-color: transparent;");
    m_vertical_measure_4_text->setPalette(palette);
    m_vertical_measure_4_text->setStyleSheet("background-color: transparent;");
    m_vertical_measure_5_text->setPalette(palette);
    m_vertical_measure_5_text->setStyleSheet("background-color: transparent;");
    m_vertical_measure_6_text->setPalette(palette);
    m_vertical_measure_6_text->setStyleSheet("background-color: transparent;");

    m_horizontal_measure_1_text->move((max_width / 2) + (max_width / 8) + 5, (max_height / 2));
    m_horizontal_measure_2_text->move((max_width / 2) + (max_width / 8)*2 + 5, (max_height / 2));
    m_horizontal_measure_3_text->move((max_width / 2) + (max_width / 8)*3 + 5, (max_height / 2));
    m_horizontal_measure_4_text->move((max_width / 2) - (max_width / 8) + 5, (max_height / 2));
    m_horizontal_measure_5_text->move((max_width / 2) - (max_width / 8)*2 + 5, (max_height / 2));
    m_horizontal_measure_6_text->move((max_width / 2) - (max_width / 8)*3 + 5, (max_height / 2));

    m_vertical_measure_1_text->move((max_width / 2) + 5, (max_height / 2) + (max_height / 8));
    m_vertical_measure_2_text->move((max_width / 2) + 5, (max_height / 2) + (max_height / 8)*2);
    m_vertical_measure_3_text->move((max_width / 2) + 5, (max_height / 2) + (max_height / 8)*3);
    m_vertical_measure_4_text->move((max_width / 2) + 5, (max_height / 2) - (max_height / 8));
    m_vertical_measure_5_text->move((max_width / 2) + 5, (max_height / 2) - (max_height / 8)*2);
    m_vertical_measure_6_text->move((max_width / 2) + 5, (max_height / 2) - (max_height / 8)*3);
}

void DrawManager::getTrackPosition(int &pos_x, int &pos_y, double distance, double bearing)
{
    int max_width = m_main_window.m_view_widget_ptr.get()->width();
    int max_height = m_main_window.m_view_widget_ptr.get()->height();

    bearing = (bearing - 90);

    double center_x = max_width / 2;
    double center_y = max_height / 2;

    double tmp_pos_x = distance * cos(M_PI * bearing / 180.0);
    double tmp_pos_y = distance * sin(M_PI * bearing / 180.0);

    double max_range = ConfigurationManager::getInstance()->getDrawConfig().draw_range;
    pos_x = center_x + (tmp_pos_x * (m_main_window.m_view_widget_ptr.get()->width() / 2) / max_range);
    pos_y = center_y + (tmp_pos_y * (m_main_window.m_view_widget_ptr.get()->height() / 2) / max_range);
}

void DrawManager::addHeadingBar(Track *track, QLabel *label, QPixmap &base_pixmap)
{
    // Add heading bar
    QPixmap heading_bar;
    if (track->getBehavior() == Track::TrackBehavior::ALLIED)
    {
        heading_bar = blue_bar_mid_img;
    }
    else if (track->getBehavior() == Track::TrackBehavior::HOSTILE)
    {
        heading_bar = red_bar_mid_img;
    }
    else
    {
        heading_bar = yellow_bar_mid_img;
    }
    int heading_new_width = heading_bar.width() * m_track_scale;
    int heading_new_height = heading_bar.height() * m_track_scale;
    QPixmap scaled_heading_bar = heading_bar.scaled(heading_new_width, heading_new_height, Qt::KeepAspectRatio);

    // Calculate combined dimensions
    int combined_width = base_pixmap.width() + scaled_heading_bar.height() + 7;
    int combined_height = base_pixmap.height() + scaled_heading_bar.height() + 10;

    QPixmap combinedPixmap(combined_width, combined_height);
    combinedPixmap.fill(Qt::transparent);
    //combinedPixmap.fill(Qt::black);

    QPainter painter(&combinedPixmap);
    painter.drawPixmap((combined_width - base_pixmap.width()) / 2, (combined_height - base_pixmap.height()) / 2, base_pixmap);


    // Calculate rotation point and apply rotation matrix
    QPointF rotationPoint(0, 0);
    QTransform matrix;
    matrix.translate(rotationPoint.x(), rotationPoint.y());
    matrix.rotate(track->getHeading());
    matrix.translate(-rotationPoint.x(), -rotationPoint.y());

    QPixmap rotatedPixmap = scaled_heading_bar.transformed(matrix);

    int overlayX = ((combined_width - rotatedPixmap.width()) / 2) + (rotatedPixmap.width() / 2 + 7) * sin(M_PI * track->getHeading() / 180.0);
    int overlayY = ((combined_height - rotatedPixmap.height()) / 2) - (rotatedPixmap.height() / 2 + 10) * cos(M_PI * track->getHeading() / 180.0);

    painter.drawPixmap(overlayX, overlayY, rotatedPixmap);
    painter.end();

    QRect geometry_selector = label->geometry();
    geometry_selector.setWidth(combinedPixmap.width());
    geometry_selector.setHeight(combinedPixmap.height());
    label->setGeometry(geometry_selector);

    base_pixmap = combinedPixmap;

    label->setPixmap(combinedPixmap);
}

QPixmap DrawManager::getTrackImage(Track *track)
{
    Track::TrackType type = track->getType();
    Track::TrackBehavior behavior = track->getBehavior();
    QPixmap res;

    if (type == Track::TrackType::AIR)
    {
        if (behavior == Track::TrackBehavior::ALLIED)
        {
            res = air_ally_img;
        }
        else if (behavior == Track::TrackBehavior::HOSTILE)
        {
            res = air_enemy_img;
        }
        else
        {
            res = air_unknown_img;
        }
    }
    else if (type == Track::TrackType::SURFACE)
    {
        if (behavior == Track::TrackBehavior::ALLIED)
        {
            res = surface_ally_img;
        }
        else if (behavior == Track::TrackBehavior::HOSTILE)
        {
            res = surface_enemy_img;
        }
        else
        {
            res = surface_unknown_img;
        }
    }
    else
    {
        res = surface_unknown_img;
    }

    return res;
}

void DrawManager::drawTracks()
{
    for (int i=0; i < m_tracker->getTracks()->keys().size(); i++)
    {
        ClickableLabel *label = m_label_map[i];
        QLabel *infolabel = m_info_label_map[i];
        int track_id = m_tracker->getTracks()->keys()[i];
        Track *track = m_tracker->getTracks()->value(track_id);
        if (track)
        {
            int newWidth = getTrackImage(track).width() * m_track_scale;
            int newHeight = getTrackImage(track).height() * m_track_scale;
            QPixmap scaledPixmap = getTrackImage(track).scaled(newWidth, newHeight, Qt::KeepAspectRatio);
            label->setPixmap(scaledPixmap);

            QRect currentGeometry = label->geometry();
            currentGeometry.setWidth(scaledPixmap.width());
            currentGeometry.setHeight(scaledPixmap.height());
            label->setGeometry(currentGeometry);

            addHeadingBar(track, label, scaledPixmap);

            int pos_x;
            int pos_y;
            getTrackPosition(pos_x, pos_y, track->getDistance(), track->getBearing());
            int position_x = pos_x - label->pixmap().width() / 2;
            int position_y = pos_y - label->pixmap().height() / 2;
            if (position_x < ConfigurationManager::getInstance()->getWindowConfig().view_window_width &&
                position_y < ConfigurationManager::getInstance()->getWindowConfig().view_window_height)
            {
                label->move(position_x, position_y);
                label->setVisible(true);
                label->setAttribute(Qt::WA_TranslucentBackground);
            }
            else
            {
                label->setVisible(false);
            }

            connect(label, &ClickableLabel::clicked, [this, track] {
                trackSelected(track);
            });

            QPainter painter(&scaledPixmap);

            if (m_selected_track_id == track->getId())
            {
                if (track->getType() == Track::TrackType::SURFACE || track->getType() == Track::TrackType::UNKNOWN)
                {
                    int selector_new_width = surface_selector_img.width() * m_track_scale;
                    int selector_new_height = surface_selector_img.height() * m_track_scale;
                    QPixmap scaledSelector = surface_selector_img.scaled(selector_new_width, selector_new_height, Qt::KeepAspectRatio);
                    int overlayY = (scaledPixmap.height() - scaledSelector.height()) / 2;
                    int overlayX = (scaledPixmap.width() - scaledSelector.width()) / 2;

                    painter.drawPixmap(overlayX, overlayY, scaledSelector);
                    label->setPixmap(scaledPixmap);
                }
                else if (track->getType() == Track::TrackType::AIR)
                {
                    int selector_new_width = air_selector_img.width() * m_track_scale;
                    int selector_new_height = air_selector_img.height() * m_track_scale;
                    QPixmap scaledSelector = air_selector_img.scaled(selector_new_width, selector_new_height, Qt::KeepAspectRatio);
                    int overlayY = (scaledPixmap.height() - scaledSelector.height()) / 2;
                    int overlayX = (scaledPixmap.width() - scaledSelector.width()) / 2;

                    painter.drawPixmap(overlayX, overlayY, scaledSelector);
                    label->setPixmap(scaledPixmap);

                    int adjusted_pos_x;
                    int adjusted_pos_y;
                    getTrackPosition(adjusted_pos_x, adjusted_pos_y, track->getDistance(), track->getBearing());

                    if (position_x < ConfigurationManager::getInstance()->getWindowConfig().view_window_width &&
                        position_y < ConfigurationManager::getInstance()->getWindowConfig().view_window_height)
                    {
                        label->move(adjusted_pos_x - scaledPixmap.width() / 2, adjusted_pos_y - scaledPixmap.height() / 2);
                    }
                    else
                    {
                        label->setVisible(false);
                    }
                }
                painter.end();
            }
            if (m_show_track_info)
            {
                QFont font;
                font.setPointSize(9);
                //font.setBold(true);

                QPen textPen;
                if (track->getBehavior() == Track::TrackBehavior::ALLIED)
                {
                    textPen.setColor(QColor(0, 168, 236));
                }
                else if (track->getBehavior() == Track::TrackBehavior::HOSTILE)
                {
                    textPen.setColor(QColor(255, 0, 0));
                }
                else
                {
                   textPen.setColor(QColor(248, 244, 0));
                }

                QString text_1 = "DST: " + QString::number(track->getDistance(), 'f', 2) +
                                 " BRG: " + QString::number(track->getBearing(), 'f', 0);

                QString text_2 = "HDG: " + QString::number(track->getHeading(), 'f', 0) +
                                 " SPD: " + QString::number(track->getSpeed(), 'f', 1);

                QString combined_text = text_1 + " " + text_2;

                QFontMetrics fontMetrics(font);
                int textWidth = fontMetrics.horizontalAdvance(combined_text);
                int textHeight = fontMetrics.height() - 1;

                QPixmap info_pixmap(textWidth, textHeight);
                info_pixmap.fill(Qt::transparent);

                QPainter info_painter(&info_pixmap);

                info_painter.setFont(font);
                info_painter.setPen(textPen);
                info_painter.drawText(QPoint(0, textHeight), combined_text);
                info_painter.end();

                infolabel->setPixmap(info_pixmap);
                infolabel->setFixedSize(textWidth, textHeight);
                infolabel->setWordWrap(true);
                infolabel->setAttribute(Qt::WA_TranslucentBackground);
                infolabel->move((label->pos().x() + scaledPixmap.width() / 2) - (info_pixmap.width() / 2), label->pos().y() + scaledPixmap.height() - 1);
                infolabel->setVisible(true);
            }
            else
            {
                infolabel->setVisible(false);
            }
        }
        else
        {
            // Hide label
            if (label->isVisible())
            {
                label->setVisible(false);
                if (m_selected_track_id == track_id)
                {
                    unselecTrack();
                }
            }

            if (infolabel->isVisible())
            {
                infolabel->setVisible(false);
            }
        }
    }
    m_main_window.update();
    //m_main_window.repaint();
}

void DrawManager::sliderValueChanged(int new_value)
{
    double range = new_value * m_step_dist;
    BOOST_LOG_SEV(m_logger, debug) << "Range slider value changed to " << range << " m";

    ConfigurationManager::getInstance()->setRange(range);

    double h_measure_1_value = (range / 4.) * 1;
    double h_measure_2_value = (range / 4.) * 2;
    double h_measure_3_value = (range / 4.) * 3;
    double h_measure_4_value = (range / 4.) * 1;
    double h_measure_5_value = (range / 4.) * 2;
    double h_measure_6_value = (range / 4.) * 3;

    double v_measure_1_value = (range / 4.) * 1;
    double v_measure_2_value = (range / 4.) * 2;
    double v_measure_3_value = (range / 4.) * 3;
    double v_measure_4_value = (range / 4.) * 1;
    double v_measure_5_value = (range / 4.) * 2;
    double v_measure_6_value = (range / 4.) * 3;

    QString h_measure_1_value_text = std::fmod(h_measure_1_value, 1.0) == 0.0 ? QString::number(h_measure_1_value, 'f', 0) : QString::number(h_measure_1_value, 'f', 2);
    QString h_measure_2_value_text = std::fmod(h_measure_2_value, 1.0) == 0.0 ? QString::number(h_measure_2_value, 'f', 0) : QString::number(h_measure_2_value, 'f', 2);
    QString h_measure_3_value_text = std::fmod(h_measure_3_value, 1.0) == 0.0 ? QString::number(h_measure_3_value, 'f', 0) : QString::number(h_measure_3_value, 'f', 2);
    QString h_measure_4_value_text = std::fmod(h_measure_4_value, 1.0) == 0.0 ? QString::number(h_measure_4_value, 'f', 0) : QString::number(h_measure_4_value, 'f', 2);
    QString h_measure_5_value_text = std::fmod(h_measure_5_value, 1.0) == 0.0 ? QString::number(h_measure_5_value, 'f', 0) : QString::number(h_measure_5_value, 'f', 2);
    QString h_measure_6_value_text = std::fmod(h_measure_6_value, 1.0) == 0.0 ? QString::number(h_measure_6_value, 'f', 0) : QString::number(h_measure_6_value, 'f', 2);

    QString v_measure_1_value_text = std::fmod(v_measure_1_value, 1.0) == 0.0 ? QString::number(v_measure_1_value, 'f', 0) : QString::number(v_measure_1_value, 'f', 2);
    QString v_measure_2_value_text = std::fmod(v_measure_2_value, 1.0) == 0.0 ? QString::number(v_measure_2_value, 'f', 0) : QString::number(v_measure_2_value, 'f', 2);
    QString v_measure_3_value_text = std::fmod(v_measure_3_value, 1.0) == 0.0 ? QString::number(v_measure_3_value, 'f', 0) : QString::number(v_measure_3_value, 'f', 2);
    QString v_measure_4_value_text = std::fmod(v_measure_4_value, 1.0) == 0.0 ? QString::number(v_measure_4_value, 'f', 0) : QString::number(v_measure_4_value, 'f', 2);
    QString v_measure_5_value_text = std::fmod(v_measure_5_value, 1.0) == 0.0 ? QString::number(v_measure_5_value, 'f', 0) : QString::number(v_measure_5_value, 'f', 2);
    QString v_measure_6_value_text = std::fmod(v_measure_6_value, 1.0) == 0.0 ? QString::number(v_measure_6_value, 'f', 0) : QString::number(v_measure_6_value, 'f', 2);

    m_horizontal_measure_1_text->setText(h_measure_1_value_text);
    m_horizontal_measure_2_text->setText(h_measure_2_value_text);
    m_horizontal_measure_3_text->setText(h_measure_3_value_text + " m");
    m_horizontal_measure_4_text->setText(h_measure_4_value_text);
    m_horizontal_measure_5_text->setText(h_measure_5_value_text);
    m_horizontal_measure_6_text->setText(h_measure_6_value_text);

    m_vertical_measure_1_text->setText(v_measure_1_value_text);
    m_vertical_measure_2_text->setText(v_measure_2_value_text);
    m_vertical_measure_3_text->setText(v_measure_3_value_text);
    m_vertical_measure_4_text->setText(v_measure_4_value_text);
    m_vertical_measure_5_text->setText(v_measure_5_value_text);
    m_vertical_measure_6_text->setText(v_measure_6_value_text + " m");

    m_horizontal_measure_1_text->adjustSize();
    m_horizontal_measure_2_text->adjustSize();
    m_horizontal_measure_3_text->adjustSize();
    m_horizontal_measure_4_text->adjustSize();
    m_horizontal_measure_5_text->adjustSize();
    m_horizontal_measure_6_text->adjustSize();

    m_vertical_measure_1_text->adjustSize();
    m_vertical_measure_2_text->adjustSize();
    m_vertical_measure_3_text->adjustSize();
    m_vertical_measure_4_text->adjustSize();
    m_vertical_measure_5_text->adjustSize();
    m_vertical_measure_6_text->adjustSize();

    m_main_window.update();
}

void DrawManager::checkboxValueChanged(int new_state)
{
    bool show_track_info = new_state == 0 ? false : true;
    BOOST_LOG_SEV(m_logger, debug) << "Show track info checkbox value changed from: " << m_show_track_info << " to: " << show_track_info;
    m_show_track_info = show_track_info;
}

void DrawManager::trackSelected(Track *track)
{
    if (track)
    {
        BOOST_LOG_SEV(m_logger, debug) << "Selected track with id: " << track;

        QPalette palette = m_track_distance_value->palette();
        palette.setColor(QPalette::Text, Qt::black);

        m_track_distance_value->setText(QString::number(track->getDistance()));
        m_track_distance_value->setPalette(palette);

        m_track_bearing_value->setText(QString::number(track->getBearing()));
        m_track_bearing_value->setPalette(palette);

        m_track_heading_value->setText(QString::number(track->getHeading()));
        m_track_heading_value->setPalette(palette);

        m_track_speed_value->setText(QString::number(track->getSpeed()));
        m_track_speed_value->setPalette(palette);

        m_track_type_value->setText(track->getTypeStr());
        m_track_type_value->setPalette(palette);

        m_track_behavior_value->setText(track->getBehaviorStr());
        m_track_behavior_value->setPalette(palette);

        m_selected_track_id = track->getId();
    }
}

void DrawManager::unselecTrack()
{
    BOOST_LOG_SEV(m_logger, debug) << "Unselected track with id: " << m_selected_track_id;
    m_selected_track_id = -1;

    QPalette palette = m_track_distance_value->palette();
    palette.setColor(QPalette::Text, Qt::gray);

    m_track_distance_value->setText("No track selected");
    m_track_distance_value->setPalette(palette);

    m_track_bearing_value->setText("No track selected");
    m_track_bearing_value->setPalette(palette);

    m_track_heading_value->setText("No track selected");
    m_track_heading_value->setPalette(palette);

    m_track_speed_value->setText("No track selected");
    m_track_speed_value->setPalette(palette);

    m_track_type_value->setText("No track selected");
    m_track_type_value->setPalette(palette);

    m_track_behavior_value->setText("No track selected");
    m_track_behavior_value->setPalette(palette);
}

void DrawManager::updateSelectedTrackInfo()
{
    for (int i=0; i < m_tracker->getTracks()->keys().size(); i++)
    {
        int track_id = m_tracker->getTracks()->keys()[i];
        Track *track = m_tracker->getTracks()->value(track_id);
        if (track && track_id == m_selected_track_id)
        {
            QPalette palette = m_track_distance_value->palette();
            palette.setColor(QPalette::Text, Qt::black);

            m_track_distance_value->setText(QString::number(track->getDistance()));
            m_track_distance_value->setPalette(palette);

            m_track_bearing_value->setText(QString::number(track->getBearing()));
            m_track_bearing_value->setPalette(palette);

            m_track_heading_value->setText(QString::number(track->getHeading()));
            m_track_heading_value->setPalette(palette);

            m_track_speed_value->setText(QString::number(track->getSpeed()));
            m_track_speed_value->setPalette(palette);

            m_track_type_value->setText(track->getTypeStr());
            m_track_type_value->setPalette(palette);

            m_track_behavior_value->setText(track->getBehaviorStr());
            m_track_behavior_value->setPalette(palette);
        }
    }
}
