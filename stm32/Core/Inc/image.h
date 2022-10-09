#ifndef __Image_H
#define __Image_H

#include <stdint.h>

enum class ColorOrder
{
    IMAGE_BIG_END,
    IMAGE_SMALL_END
};

class Image
{
public:
    explicit Image(uint16_t width, uint16_t height, ColorOrder order, const unsigned char* data)
        : width_(width), height_(height), order_(order), data_(data)
    {
    }

    uint16_t width() const
    {
        return width_;
    }

    uint16_t height() const
    {
        return height_;
    }

    ColorOrder order() const
    {
        return order_;
    }

    /** @brief 获取颜色
     * @param x 像素的横坐标
     * @param y 像素的纵坐标
     */
    uint16_t colorAt(uint16_t x, uint16_t y)
    {
        return colorAt(x * width_ + y);
    }

    /** @brief 获取颜色
     * @param index 像素的索引值
     */
    uint16_t colorAt(uint16_t index)
    {
        uint32_t pos = index * 2;
        if (order_ == ColorOrder::IMAGE_SMALL_END)
            return (data_[pos + 1] << 8) | data_[pos];
        else
            return (data_[pos] << 8) | data_[pos + 1];
    }

private:
    uint16_t width_;
    uint16_t height_;
    ColorOrder order_;
    const unsigned char* data_;
};

#endif