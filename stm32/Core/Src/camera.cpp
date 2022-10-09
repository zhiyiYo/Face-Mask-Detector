#include "camera.h"
#include "stdio.h"

#include "ov7725config.h"

CameraTypeDef cameraTypeDef;
extern uint8_t numBufferImages;

Camera::Camera(LCD* lcd) : lcd_(lcd)
{
    uint16_t reg = 0;

    if (!sccb_.writeRegister(0x12, 0x80))
    {
        printf("Ov7725 Reset failded\r\n");
        return;
    }

    //读取厂家ID 高八位
    HAL_Delay(50);
    reg = sccb_.readRegister(0X1c);
    reg <<= 8;
    reg |= sccb_.readRegister(0X1d);
    if (reg != MID)
    {
        printf("MID:%d\r\n", reg);
        return;
    }

    //读取厂家ID
    reg = sccb_.readRegister(0X0a);
    reg <<= 8;
    reg |= sccb_.readRegister(0X0b);
    if (reg != PID)
    {
        printf("HID:%d\r\n", reg);
        return;
    }

    //初始化 OV7725,采用 QVGA 分辨率(320*240)
    for (auto &[reg, value] : cameraConfig)
        sccb_.writeRegister(reg, value);

    setWindow(WIDTH, HEIGHT, ImageMode::QVGA);
    setLightMode(CameraLightMode::AUTO);
    setSaturation(0);
    setBrightness(0);
    setContrast(0);
    cameraTypeDef.CS = 0;
}

/** @brief 设置白平衡模式
 * @param mode 白平衡模式
 */
void Camera::setLightMode(CameraLightMode mode)
{
    switch (mode)
    {
        case CameraLightMode::AUTO:
            sccb_.writeRegister(0x13, 0xff);  // AWB on
            sccb_.writeRegister(0x0e, 0x65);
            sccb_.writeRegister(0x2d, 0x00);
            sccb_.writeRegister(0x2e, 0x00);
            break;
        case CameraLightMode::SUNNY:
            sccb_.writeRegister(0x13, 0xfd);  // AWB off
            sccb_.writeRegister(0x01, 0x5a);
            sccb_.writeRegister(0x02, 0x5c);
            sccb_.writeRegister(0x0e, 0x65);
            sccb_.writeRegister(0x2d, 0x00);
            sccb_.writeRegister(0x2e, 0x00);
            break;
        case CameraLightMode::CLOUDY:
            sccb_.writeRegister(0x13, 0xfd);  // AWB off
            sccb_.writeRegister(0x01, 0x58);
            sccb_.writeRegister(0x02, 0x60);
            sccb_.writeRegister(0x0e, 0x65);
            sccb_.writeRegister(0x2d, 0x00);
            sccb_.writeRegister(0x2e, 0x00);
            break;
        case CameraLightMode::OFFICE:
            sccb_.writeRegister(0x13, 0xfd);  // AWB off
            sccb_.writeRegister(0x01, 0x84);
            sccb_.writeRegister(0x02, 0x4c);
            sccb_.writeRegister(0x0e, 0x65);
            sccb_.writeRegister(0x2d, 0x00);
            sccb_.writeRegister(0x2e, 0x00);
            break;
        case CameraLightMode::HOME:
            sccb_.writeRegister(0x13, 0xfd);  // AWB off
            sccb_.writeRegister(0x01, 0x96);
            sccb_.writeRegister(0x02, 0x40);
            sccb_.writeRegister(0x0e, 0x65);
            sccb_.writeRegister(0x2d, 0x00);
            sccb_.writeRegister(0x2e, 0x00);
            break;
        case CameraLightMode::NIGHT:
            sccb_.writeRegister(0x13, 0xff);  // AWB on
            sccb_.writeRegister(0x0e, 0xe5);
            break;
    }
}

/** @brief 设置饱和度
 * @param saturation 饱和度偏差值，取值范围 [-4, 4]
 */
void Camera::setSaturation(int8_t saturation)
{
    if (saturation < -4 || saturation > 4)
        return;

    sccb_.writeRegister(USAT, (saturation + 4) << 4);
    sccb_.writeRegister(VSAT, (saturation + 4) << 4);
}

/** @brief 设置亮度
 * @param bright 亮度偏差量，取值范围 [-4, 4]
 */
void Camera::setBrightness(int8_t bright)
{
    bright = bright >= 0 ? bright * 16 + 8 : (-bright - 1) * 16 + 8;
    sccb_.writeRegister(BRIGHT, bright);
    sccb_.writeRegister(SIGN, bright >= 0 ? 0x06 : 0x0e);
}

/** @brief 设置对比度
 * @param contrast 对比度偏差值，取值范围 [-4, 4]
 */
void Camera::setContrast(int8_t contrast)
{
    if (contrast >= -4 && contrast <= 4)
        sccb_.writeRegister(CNST, (0x30 - (4 - contrast) * 4));
}

//特效设置
// 0:普通模式
// 1,负片
// 2,黑白
// 3,偏红色
// 4,偏绿色
// 5,偏蓝色
// 6,复古
void Camera::setEffect(CameraEffect effect)
{
    switch (effect)
    {
        case CameraEffect::NORMAL:
            sccb_.writeRegister(0xa6, 0x06);  // TSLB设置
            sccb_.writeRegister(0x60, 0x80);  // MANV,手动V值
            sccb_.writeRegister(0x61, 0x80);  // MANU,手动U值
            break;
        case CameraEffect::NEGATIVE: sccb_.writeRegister(0xa6, 0x46); break;
        case CameraEffect::GREY:
            sccb_.writeRegister(0xa6, 0x26);
            sccb_.writeRegister(0x60, 0x80);
            sccb_.writeRegister(0x61, 0x80);
            break;
        case CameraEffect::RED:
            sccb_.writeRegister(0xa6, 0x1e);
            sccb_.writeRegister(0x60, 0x80);
            sccb_.writeRegister(0x61, 0xc0);
            break;
        case CameraEffect::GREEN:
            sccb_.writeRegister(0xa6, 0x1e);
            sccb_.writeRegister(0x60, 0x60);
            sccb_.writeRegister(0x61, 0x60);
            break;
        case CameraEffect::BLUE:
            sccb_.writeRegister(0xa6, 0x1e);
            sccb_.writeRegister(0x60, 0xa0);
            sccb_.writeRegister(0x61, 0x40);
            break;
        case CameraEffect::RETRO:
            sccb_.writeRegister(0xa6, 0x1e);
            sccb_.writeRegister(0x60, 0x40);
            sccb_.writeRegister(0x61, 0xa0);
            break;
        default: break;
    }
}

/** @brief 设置图像输出窗口
 * @param width 输出图像宽度，不大于 320
 * @param height 输出图像高度，不大于 240
 * @param mode 输出图像模式，QVGA模式可视范围广但近物不是很清晰，VGA模式可视范围小近物清晰
 */
void Camera::setWindow(uint16_t width, uint16_t height, ImageMode mode)
{
    uint16_t sx, sy;
    if (mode == ImageMode::VGA)
    {
        sx = (640 - width) / 2;
        sy = (480 - height) / 2;
        sccb_.writeRegister(COM7, 0x06);    //设置为VGA模式
        sccb_.writeRegister(HSTART, 0x23);  //水平起始位置
        sccb_.writeRegister(HSIZE, 0xA0);   //水平尺寸
        sccb_.writeRegister(VSTRT, 0x07);   //垂直起始位置
        sccb_.writeRegister(VSIZE, 0xF0);   //垂直尺寸
        sccb_.writeRegister(HREF, 0x00);
        sccb_.writeRegister(HOutSize, 0xA0);  //输出尺寸
        sccb_.writeRegister(VOutSize, 0xF0);  //输出尺寸
    }
    else
    {
        sx = (320 - width) / 2;
        sy = (240 - height) / 2;
        sccb_.writeRegister(COM7, 0x46);  //设置为QVGA模式
        sccb_.writeRegister(HSTART, 0x3f);
        sccb_.writeRegister(HSIZE, 0x50);
        sccb_.writeRegister(VSTRT, 0x03);
        sccb_.writeRegister(VSIZE, 0x78);
        sccb_.writeRegister(HREF, 0x00);
        sccb_.writeRegister(HOutSize, 0x50);
        sccb_.writeRegister(VOutSize, 0x78);
    }

    uint8_t raw = sccb_.readRegister(HSTART);
    uint8_t temp = raw + (sx >> 2);  // sx高8位存在HSTART,低2位存在HREF[5:4]
    sccb_.writeRegister(HSTART, temp);
    sccb_.writeRegister(HSIZE, width >> 2);  // width高8位存在HSIZE,低2位存在HREF[1:0]

    raw = sccb_.readRegister(VSTRT);
    temp = raw + (sy >> 1);  // sy高8位存在VSTRT,低1位存在HREF[6]
    sccb_.writeRegister(VSTRT, temp);
    sccb_.writeRegister(VSIZE, height >> 1);  // height高8位存在VSIZE,低1位存在HREF[2]

    raw = sccb_.readRegister(HREF);
    temp = ((sy & 0x01) << 6) | ((sx & 0x03) << 4) | ((height & 0x01) << 2) | (width & 0x03) | raw;
    sccb_.writeRegister(HREF, temp);

    sccb_.writeRegister(HOutSize, width >> 2);
    sccb_.writeRegister(VOutSize, height >> 1);

    sccb_.readRegister(EXHCH);
    temp = (raw | (width & 0x03) | ((height & 0x01) << 2));
    sccb_.writeRegister(EXHCH, temp);
}

/** @brief 将 FIFO 芯片中的图像显示到 LCD 上
 */
void Camera::refresh()
{
    if (!numBufferImages)
        return;

    lcd_->setScanDirection(LCDScanDirection::U2D_L2R);
    lcd_->startDraw();

    cameraTypeDef.RRST = 0;  //开始复位读指针
    setReadClock(GPIO_PIN_RESET);
    setReadClock(GPIO_PIN_SET);
    setReadClock(GPIO_PIN_RESET);

    cameraTypeDef.RRST = 1;  //复位读指针结束
    setReadClock(GPIO_PIN_SET);

    for (uint32_t i = 0; i < HEIGHT; ++i)
    {
        for (uint32_t j = 0; j < WIDTH; ++j)
        {
            // 读颜色的高八位
            setReadClock(GPIO_PIN_RESET);
            uint16_t color = GPIOC->IDR & 0XFF;
            setReadClock(GPIO_PIN_SET);

            color <<= 8;

            // 读颜色的低八位
            setReadClock(GPIO_PIN_RESET);
            color |= GPIOC->IDR & 0XFF;
            setReadClock(GPIO_PIN_SET);

            lcd_->drawPoint(color);
        }
    }

    lcd_->setScanDirection(LCDScanDirection::L2R_U2D);
    numBufferImages = 0;  //清零帧中断标记
}
