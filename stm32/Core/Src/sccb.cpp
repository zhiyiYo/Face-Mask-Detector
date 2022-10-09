#include "sccb.h"
#include "delay.h"

SCCB::SCCB() : SCL_(PDout(13)), SDA_(PGout(13)), readSDA_(PGin(13))
{
    setSDADirection(SDADirection::OUT);
}

/** @brief 开始传输
 */
void SCCB::start()
{
    SDA_ = 1;  //数据线高电平
    SCL_ = 1;  //在时钟线高的时候数据线由高至低
    delay_us(50);
    SDA_ = 0;
    delay_us(50);
    SCL_ = 0;  //数据线恢复低电平，单操作函数必要
}

/** @brief 结束传输
 */
void SCCB::stop()
{
    SDA_ = 0;
    delay_us(50);
    SCL_ = 1;
    delay_us(50);
    SDA_ = 1;
    delay_us(50);
}

/** @brief 产生 NA 信号
 */
void SCCB::noAck()
{
    delay_us(50);
    SDA_ = 1;
    SCL_ = 1;
    delay_us(50);
    SCL_ = 0;
    delay_us(50);
    SDA_ = 0;
    delay_us(50);
}

/** @brief 写入一个字节的数据
 * @param value 数据
 * @return success 传输是否成功
 */
bool SCCB::writeByte(uint8_t value)
{
    for (uint8_t i = 0; i < 8; ++i)
    {
        if (value & 0x80)
            SDA_ = 1;
        else
            SDA_ = 0;

        value <<= 1;
        delay_us(50);
        SCL_ = 1;
        delay_us(50);
        SCL_ = 0;
    }

    setSDADirection(SDADirection::IN);
    delay_us(50);

    //接收第九位,以判断是否发送成功
    SCL_ = 1;
    delay_us(50);
    bool success = !readSDA_;
    SCL_ = 0;

    setSDADirection(SDADirection::OUT);
    return success;
}

/** @brief 读取一个字节的数据
 */
uint8_t SCCB::readByte()
{
    setSDADirection(SDADirection::IN);

    uint8_t value = 0;
    for (uint8_t j = 8; j > 0; --j)
    {
        delay_us(50);
        SCL_ = 1;
        value = value << 1;
        if (readSDA_)
            value++;

        delay_us(50);
        SCL_ = 0;
    }

    setSDADirection(SDADirection::OUT);
    return value;
}

/** @brief 写寄存器
 * @param reg 寄存器
 * @param value 写入的值
 */
bool SCCB::writeRegister(uint8_t reg, uint8_t value)
{
    bool success = true;
    start();

    // 写入 OV7725 的设备 ID
    success &= writeByte(SCCB::ID);
    delay_us(100);

    //写寄存器地址
    success &= writeByte(reg);
    delay_us(100);

    //写数据
    success &= writeByte(value);

    stop();
    return success;
}

/** @brief 读寄存器
 * @param reg 寄存器
 */
uint8_t SCCB::readRegister(uint8_t reg)
{
    start();

    //写 OV7725 的设备 ID
    writeByte(SCCB::ID);
    delay_us(100);

    //写寄存器地址
    writeByte(reg);
    delay_us(100);
    stop();
    delay_us(100);

    //读数据
    start();
    writeByte(SCCB::ID | 0X01);  //发送读命令
    delay_us(100);
    uint8_t value = readByte(); //读取数据
    noAck();
    stop();

    return value;
}
