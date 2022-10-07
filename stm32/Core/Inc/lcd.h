#pragma once
#include "image.h"
#include "stdint.h"

enum class LCDDirection
{
    NORMAL,
    ROTATE_90,
    ROTATE_180,
    ROTATE_270
};

class LCD
{
public:
    enum class Command
    {
        NOP = 0x00,
        SLEEP_OUT = 0x11,
        COLUMN_ADDRESS_SET = 0x2A,
        PAGE_ADDRESS_SET = 0x2B,
        WRITE_RAM = 0x2C,
        GAMMA_SET = 0x26,
        DISPLAY_ON = 0x29,
        MEMORY_ACCESS_CONTROL = 0x36,
        PIXEL_FORMAT_SET = 0x3A,
        FRAME_RATE_CONTROL = 0xB1,
        DISPLAY_FUNCTION_CONTROL = 0xB6,
        POWER_CONTROL_1 = 0xC0,
        POWER_CONTROL_2 = 0xC1,
        VCOM_CONTROL_1 = 0xC5,
        POWER_CONTROL_A = 0xCB,
        POWER_CONTROL_B = 0xCF,
        VCOM_CONTROL_2 = 0xC7,
        READ_ID4 = 0xD3,
        POSITIVE_GAMMA_CORRECTION = 0xE0,
        NEGATIVE_GAMMA_CORRECTION = 0xE1,
        DIRVER_TIMING_CONTROL_A = 0xE8,
        DIRVER_TIMING_CONTROL_B = 0xEA,
        POWER_ON_SEQUENCE = 0xED,
        PUMP_RATIO_CONTROL = 0xF7,
        ENABLE_3G = 0xF2,
    };

    LCD(uint16_t width, uint16_t height, LCDDirection direction = LCDDirection::NORMAL);

    uint16_t id() const
    {
        return id_;
    }

    void setCursor(uint16_t x, uint16_t y);
    void setWindow(uint16_t x, uint16_t y, uint16_t width, uint16_t height);
    void setDirection(LCDDirection direction);
    void drawImage(uint16_t x, uint16_t y, Image image);
    void clear(uint16_t color);

private:
    // 驱动号
    uint16_t id_;

    // 尺寸
    uint16_t originWidth_;
    uint16_t originHeight_;
    uint16_t width_;
    uint16_t height_;

    // 显示方向
    LCDDirection direction_;

    // 要写的寄存器序号和显存数据寄存器
    uint16_t* command_;
    uint16_t* ram_;

    void setCommand(Command command);

    uint16_t getData(Command command);
    uint16_t getData();

    void setData(uint16_t value);
    void setData(Command command, uint16_t value);
};