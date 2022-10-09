#pragma once

#include "stdint.h"
#include "stm32f1xx_hal.h"
#include "sys.h"

class SCCB
{
public:
    SCCB();
    bool writeRegister(uint8_t reg, uint8_t value);
    uint8_t readRegister(uint8_t reg);

private:
    enum class SDADirection
    {
        IN = 0X00800000,
        OUT = 0X00300000
    };

    static const uint8_t ID = 0x42;

    volatile uint32_t& SCL_ = PDout(13);
    volatile uint32_t& SDA_ = PGout(13);
    volatile uint32_t& readSDA_ = PGin(13);

    void start();
    void stop();
    void noAck();

    uint8_t readByte();
    bool writeByte(uint8_t value);

    /** @brief 设置 SDA 引脚的数据流动方向
     * @param direction 方向
     */
    void setSDADirection(SDADirection direction)
    {
        GPIOG->CRH &= 0XFF0FFFFF;
        GPIOG->CRH |= static_cast<uint32_t>(direction);
    }
};