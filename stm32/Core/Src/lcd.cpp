#include "lcd.h"
#include "stdio.h"
#include "stm32f1xx_hal.h"

LCD::LCD(uint16_t width, uint16_t height, LCDDisplayDirection direction)
    : originWidth_(width), originHeight_(height), width_(width), height_(height),
      direction_(direction), lcdTypeDef_((LCDTypeDef*)((uint32_t)(0x6C000000 | 0x000007FE)))
{
    HAL_Delay(50);
    id_ = getData(Command::NOP);

    // 读到ID不正确, 9341 在未被复位的情况下会被读成 9300
    if (id_ < 0XFF || id_ == 0XFFFF || id_ == 0X9300)
    {
        setCommand(Command::READ_ID4);
        getData();        // dummy read
        getData();        // 读到0X00
        id_ = getData();  // 读取93
        id_ <<= 8;
        id_ |= getData();  //读取41
    }

    printf("LCD ID:%x\r\n", id_);  //打印 LCD ID

    setCommand(Command::POWER_CONTROL_B);
    setData(0x00);
    setData(0xC1);
    setData(0X30);

    setCommand(Command::POWER_ON_SEQUENCE);
    setData(0x64);
    setData(0x03);
    setData(0X12);
    setData(0X81);

    setCommand(Command::DIRVER_TIMING_CONTROL_A);
    setData(0x85);
    setData(0x10);
    setData(0x7A);

    setCommand(Command::POWER_CONTROL_A);
    setData(0x39);
    setData(0x2C);
    setData(0x00);
    setData(0x34);
    setData(0x02);

    setCommand(Command::PUMP_RATIO_CONTROL);
    setData(0x20);

    setCommand(Command::DIRVER_TIMING_CONTROL_B);
    setData(0x00);
    setData(0x00);

    setCommand(Command::POWER_CONTROL_1);
    setData(0x1B);  // VRH[5:0]

    setCommand(Command::POWER_CONTROL_2);
    setData(0x01);  // SAP[2:0];BT[3:0]

    setCommand(Command::VCOM_CONTROL_1);
    setData(0x30);  // 3F
    setData(0x30);  // 3C

    setCommand(Command::VCOM_CONTROL_2);
    setData(0XB7);

    setCommand(Command::MEMORY_ACCESS_CONTROL);
    setData(0x48);

    setCommand(Command::PIXEL_FORMAT_SET);
    setData(0x55);

    setCommand(Command::FRAME_RATE_CONTROL);
    setData(0x00);
    setData(0x1A);

    setCommand(Command::DISPLAY_FUNCTION_CONTROL);
    setData(0x0A);
    setData(0xA2);

    setCommand(Command::ENABLE_3G);  // 3Gamma Function Disable
    setData(0x00);

    setCommand(Command::GAMMA_SET);  // Gamma curve selected
    setData(0x01);

    setCommand(Command::POSITIVE_GAMMA_CORRECTION);
    setData(0x0F);
    setData(0x2A);
    setData(0x28);
    setData(0x08);
    setData(0x0E);
    setData(0x08);
    setData(0x54);
    setData(0XA9);
    setData(0x43);
    setData(0x0A);
    setData(0x0F);
    setData(0x00);
    setData(0x00);
    setData(0x00);
    setData(0x00);

    setCommand(Command::NEGATIVE_GAMMA_CORRECTION);
    setData(0x00);
    setData(0x15);
    setData(0x17);
    setData(0x07);
    setData(0x11);
    setData(0x06);
    setData(0x2B);
    setData(0x56);
    setData(0x3C);
    setData(0x05);
    setData(0x10);
    setData(0x0F);
    setData(0x3F);
    setData(0x3F);
    setData(0x0F);

    setCommand(Command::PAGE_ADDRESS_SET);
    setData(0x00);
    setData(0x00);
    setData(0x01);
    setData(0x3f);

    setCommand(Command::COLUMN_ADDRESS_SET);
    setData(0x00);
    setData(0x00);
    setData(0x00);
    setData(0xef);

    setCommand(Command::SLEEP_OUT);
    HAL_Delay(120);

    setCommand(Command::DISPLAY_ON);

    setDisplayDirection(direction);
    setScanDirection(LCDScanDirection::L2R_U2D);

    // 背光
    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0, GPIO_PIN_SET);
    clear(0xFFFF);
}

/** @brief 读取指定指令对应的数据
 */
uint16_t LCD::getData(Command command)
{
    setCommand(command);
    HAL_Delay(1);
    return getData();  //返回读到的值
}

/** @brief 读取当前指令对应的数据
 */
uint16_t LCD::getData()
{
    uint16_t ram;  //防止被优化
    ram = lcdTypeDef_->ram;
    return ram;
}

/** @brief 写入指令，同时写入指令的参数
 * @param command 指令
 * @param value 要写入的参数
 */
void LCD::setCommand(Command command, uint16_t value)
{
    setCommand(command);
    setData(value);
}

/** @brief 设置显示方向
 * @param direction 方向
 */
void LCD::setDisplayDirection(LCDDisplayDirection direction)
{
    direction_ = direction;
    switch (direction)
    {
        case LCDDisplayDirection::NORMAL:
            width_ = originWidth_;
            height_ = originHeight_;
            setCommand(Command::MEMORY_ACCESS_CONTROL, (1 << 3));
            break;
        case LCDDisplayDirection::ROTATE_90:
            width_ = originHeight_;
            height_ = originWidth_;
            setCommand(Command::MEMORY_ACCESS_CONTROL, (1 << 3) | (1 << 5) | (1 << 6));
            break;
        case LCDDisplayDirection::ROTATE_180:
            width_ = originWidth_;
            height_ = originHeight_;
            setCommand(Command::MEMORY_ACCESS_CONTROL, (1 << 3) | (1 << 7) | (1 << 4) | (1 << 6));
            break;
        case LCDDisplayDirection::ROTATE_270:
            width_ = originHeight_;
            height_ = originWidth_;
            setCommand(Command::MEMORY_ACCESS_CONTROL, (1 << 3) | (1 << 7) | (1 << 5) | (1 << 4));
            break;
        default: break;
    }
}

/** @brief 设置扫描方向
 * @param direction 扫描方向
 */
void LCD::setScanDirection(LCDScanDirection direction)
{
    if (direction_ == LCDDisplayDirection::ROTATE_90 ||
        direction_ == LCDDisplayDirection::ROTATE_270)
    {
        switch (direction)
        {
            case LCDScanDirection::L2R_U2D: direction = LCDScanDirection::D2U_L2R; break;
            case LCDScanDirection::L2R_D2U: direction = LCDScanDirection::D2U_R2L; break;
            case LCDScanDirection::R2L_U2D: direction = LCDScanDirection::U2D_L2R; break;
            case LCDScanDirection::R2L_D2U: direction = LCDScanDirection::U2D_R2L; break;
            case LCDScanDirection::U2D_L2R: direction = LCDScanDirection::L2R_D2U; break;
            case LCDScanDirection::U2D_R2L: direction = LCDScanDirection::L2R_U2D; break;
            case LCDScanDirection::D2U_L2R: direction = LCDScanDirection::R2L_D2U; break;
            case LCDScanDirection::D2U_R2L: direction = LCDScanDirection::R2L_U2D; break;
        }
    }

    uint16_t value = 0x08;
    switch (direction)
    {
        case LCDScanDirection::L2R_U2D: value |= (0 << 7) | (0 << 6) | (0 << 5); break;
        case LCDScanDirection::L2R_D2U: value |= (1 << 7) | (0 << 6) | (0 << 5); break;
        case LCDScanDirection::R2L_U2D: value |= (0 << 7) | (1 << 6) | (0 << 5); break;
        case LCDScanDirection::R2L_D2U: value |= (1 << 7) | (1 << 6) | (0 << 5); break;
        case LCDScanDirection::U2D_L2R: value |= (0 << 7) | (0 << 6) | (1 << 5); break;
        case LCDScanDirection::U2D_R2L: value |= (0 << 7) | (1 << 6) | (1 << 5); break;
        case LCDScanDirection::D2U_L2R: value |= (1 << 7) | (0 << 6) | (1 << 5); break;
        case LCDScanDirection::D2U_R2L: value |= (1 << 7) | (1 << 6) | (1 << 5); break;
    }

    setCommand(Command::MEMORY_ACCESS_CONTROL, value);

    // 交换宽度和高度
    if (((value & 0x20) && (width_ < height_)) || (!(value & 0x20) && (width_ > height_)))
    {
        uint16_t temp = width_;
        width_ = height_;
        height_ = temp;
    }

    setWindow(0, 0, width_, height_);
}

/** @brief 设置光标位置
 * @param x 横坐标
 * @param y 纵坐标
 */
void LCD::setCursor(uint16_t x, uint16_t y)
{
    setCommand(Command::COLUMN_ADDRESS_SET);
    setData(x >> 8);
    setData(x & 0XFF);

    setCommand(Command::PAGE_ADDRESS_SET);
    setData(y >> 8);
    setData(y & 0XFF);
}

/** @brief 将光标定位到左上角并开始绘制
 */
void LCD::startDraw()
{
    setCursor(0, 0);
    setCommand(Command::WRITE_RAM);
}

/** @brief 清屏
 * @param color 屏幕颜色
 */
void LCD::clear(uint16_t color)
{
    setCursor(0, 0);
    setCommand(Command::WRITE_RAM);

    uint32_t totalpoint = width_ * height_;
    for (uint32_t index = 0; index < totalpoint; index++)
    {
        setData(color);
    }
}

/** @brief 设置绘制区域
 * @param x 绘制区域左上角的横坐标
 * @param y 绘制区域左上角的纵坐标
 * @param width 绘制区域的宽度
 * @param height 绘制区域的高度
 */
void LCD::setWindow(uint16_t x, uint16_t y, uint16_t width, uint16_t height)
{
    uint16_t ex = x + width - 1;
    uint16_t ey = y + height - 1;

    // 设置横坐标
    setCommand(Command::COLUMN_ADDRESS_SET);
    setData(x >> 8);
    setData(x & 0XFF);
    setData(ex >> 8);
    setData(ex & 0XFF);

    // 设置纵坐标
    setCommand(Command::PAGE_ADDRESS_SET);
    setData(y >> 8);
    setData(y & 0XFF);
    setData(ey >> 8);
    setData(ey & 0XFF);
}

/** @brief 绘制图像
 * @param x 图像左上角横坐标
 * @param y 图像左上角纵坐标
 */
void LCD::drawImage(uint16_t x, uint16_t y, Image image)
{
    //窗口设置
    setWindow(x, y, image.width(), image.height());
    setCommand(Command::WRITE_RAM);

    int pixels = image.height() * image.width();
    for (int i = 0; i < pixels; i++)
    {
        setData(image.colorAt(i));
    }

    //恢复显示窗口为全屏
    setWindow(0, 0, width_ - 1, height_ - 1);
}

/** @brief 绘制一个点
 * @param x 横坐标
 * @param y 纵坐标
 * @param color 点的颜色
 */
void LCD::drawPoint(uint16_t x, uint16_t y, uint16_t color)
{
    setCursor(x, y);
    setCommand(Command::WRITE_RAM, color);
}