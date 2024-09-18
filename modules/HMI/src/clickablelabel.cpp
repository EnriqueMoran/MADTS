#include "clickablelabel.h"

void ClickableLabel::mousePressEvent(QMouseEvent *)
{
    emit clicked();
}
