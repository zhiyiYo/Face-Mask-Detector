#pragma once
#include "lcd.h"
#include "sccb.h"
#include "sys.h"

enum class CameraLightMode
{
    AUTO,
    SUNNY,
    CLOUDY,
    OFFICE,
    HOME,
    NIGHT
};

enum class CameraEffect
{
    NORMAL,
    NEGATIVE,
    GREY,
    RED,
    GREEN,
    BLUE,
    RETRO  // 复古
};

enum class ImageMode
{
    QVGA,
    VGA
};


struct CameraTypeDef
{
    volatile uint32_t& VSYNC = PAin(8);
    volatile uint32_t& WRST = PDout(6);
    volatile uint32_t& WREN = PBout(3);
    volatile uint32_t& RRST = PGout(14);
    volatile uint32_t& CS = PGout(15);
};


extern CameraTypeDef cameraTypeDef;

/** @brief Ov7725 摄像头
 */
class Camera
{
public:
    Camera(LCD *lcd);

    void setLightMode(CameraLightMode mode);
    void setEffect(CameraEffect effect);
    void setSaturation(int8_t saturation);
    void setBrightness(int8_t bright);
    void setContrast(int8_t contrast);
    void setWindow(uint16_t width, uint16_t height, ImageMode mode);
    void refresh();

private:
    static const uint32_t MID = 0x7FA2;
    static const uint32_t FID = 0x7721;
    static const uint32_t WIDTH = 320;
    static const uint32_t HEIGHT = 240;

    SCCB sccb_;
    LCD* lcd_;

    /** @brief 设置读数据时钟的电平
     * @param state 时钟的电平
     */
    void setReadClock(GPIO_PinState state)
    {
        if (state == GPIO_PIN_SET)
            GPIOB->BSRR = 1 << 4;
        else
            GPIOB->BRR = 1 << 4;
    }
};

#pragma once

