#pragma once
#include "image.h"
#include "stdint.h"

enum class LCDDisplayDirection
{
    NORMAL,
    ROTATE_90,
    ROTATE_180,
    ROTATE_270
};

enum class LCDScanDirection
{
    L2R_U2D = 0,
    L2R_D2U = 1,
    R2L_U2D = 2,
    R2L_D2U = 3,
    U2D_L2R = 4,
    U2D_R2L = 5,
    D2U_L2R = 6,
    D2U_R2L = 7
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

    LCD(uint16_t width,
        uint16_t height,
        LCDDisplayDirection direction = LCDDisplayDirection::NORMAL);

    uint16_t id() const { return id_; }

    uint16_t width() const { return width_; }

    uint16_t height() const { return height_; }

    /** @brief 绘制一个点
     * @param color 点的颜色
     */
    void drawPoint(uint16_t color) { lcdTypeDef_->ram = color; }

    void setCursor(uint16_t x, uint16_t y);
    void setWindow(uint16_t x, uint16_t y, uint16_t width, uint16_t height);
    void setDisplayDirection(LCDDisplayDirection direction);
    void setScanDirection(LCDScanDirection direction);

    void startDraw();

    void clear(uint16_t color);
    void drawImage(uint16_t x, uint16_t y, Image image);
    void drawPoint(uint16_t x, uint16_t y, uint16_t color);
    void drawChar(uint16_t x, uint16_t y, uint16_t color);

private:
    // 驱动号
    uint16_t id_;

    // 尺寸
    uint16_t originWidth_;
    uint16_t originHeight_;
    uint16_t width_;
    uint16_t height_;

    // 显示方向
    LCDDisplayDirection direction_;

    // 要写的寄存器序号和显存数据寄存器
    struct LCDTypeDef
    {
        volatile uint16_t command;
        volatile uint16_t ram;
    };
    LCDTypeDef* lcdTypeDef_;

    /** @brief 设置指令
     * @param command 指令
     */
    void setCommand(Command command) { lcdTypeDef_->command = static_cast<uint16_t>(command); }

    /** @brief 写入数据
     * @param value 要写入的值
     */
    void setData(uint16_t value) { lcdTypeDef_->ram = value; }

    uint16_t getData(Command command);
    uint16_t getData();

    void setCommand(Command command, uint16_t value);
};