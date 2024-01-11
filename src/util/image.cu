#include "image.h"
#include <png.h>
#include <cstdlib>
#include <cstdarg>
#include <cstring>

void abort_(const char *s, ...) {
    va_list args;
    va_start(args, s);
    vfprintf(stderr, s, args);
    fprintf(stderr, "\n");
    va_end(args);
    abort();
}

void Image::writePNG(const std::string &file_name) {
    std::vector<png_byte> byteData(width * height * 3);
    auto ptr = byteData.begin();
    for (size_t i = 0; i < width * height; ++i) {
        Color v = pixels[i].clamp();
        *ptr++ = (unsigned char)(v.r * 255);
        *ptr++ = (unsigned char)(v.g * 255);
        *ptr++ = (unsigned char)(v.b * 255);
    }

    std::vector<png_byte *> rowData(height);
    for (int i = 0; i < height; i++)
        rowData[i] = i * width * 3 + &byteData.front();

    /* create file */
    FILE *fp = fopen(file_name.c_str(), "wb");
    if (!fp)
        abort_("[write_png_file] File %s could not be opened for writing", file_name.c_str());

    /* initialize stuff */
    png_structp png_ptr;
    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    if (!png_ptr)
        abort_("[write_png_file] png_create_write_struct failed");

    png_infop info_ptr;
    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
        abort_("[write_png_file] png_create_info_struct failed");

    if (setjmp(png_jmpbuf(png_ptr)))
        abort_("[write_png_file] Error during init_io");

    png_init_io(png_ptr, fp);

    /* write header */
    if (setjmp(png_jmpbuf(png_ptr)))
        abort_("[write_png_file] Error during writing header");

    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    png_write_info(png_ptr, info_ptr);

    /* write bytes */
    if (setjmp(png_jmpbuf(png_ptr)))
        abort_("[write_png_file] Error during writing bytes");

    png_write_image(png_ptr, (png_byte **)&rowData.front());

    /* end write */
    if (setjmp(png_jmpbuf(png_ptr)))
        abort_("[write_png_file] Error during end of write");

    png_write_end(png_ptr, NULL);

    fclose(fp);
}

void Image::readPNG(const std::string &file_name) {
    png_byte header[8]; // 8 is the maximum size that can be checked

    /* open file and test for it being a png */
    FILE *fp = fopen(file_name.c_str(), "rb");
    if (!fp)
        abort_("[read_png_file] File %s could not be opened for reading", file_name.c_str());
    fread(header, 1, 8, fp);
    if (png_sig_cmp(header, 0, 8))
        abort_("[read_png_file] File %s is not recognized as a PNG file", file_name.c_str());

    /* initialize stuff */
    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    if (!png_ptr)
        abort_("[read_png_file] png_create_read_struct failed");

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
        abort_("[read_png_file] png_create_info_struct failed");

    if (setjmp(png_jmpbuf(png_ptr)))
        abort_("[read_png_file] Error during init_io");

    png_init_io(png_ptr, fp);
    png_set_sig_bytes(png_ptr, 8);

    png_read_info(png_ptr, info_ptr);

    create(png_get_image_width(png_ptr, info_ptr), png_get_image_height(png_ptr, info_ptr));

    png_read_update_info(png_ptr, info_ptr);

    png_uint_32 rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    std::vector<png_byte> byteData(rowbytes * height);
    std::vector<png_byte *> rowData(height);
    for (int i = 0; i < height; i++)
        rowData[i] = i * rowbytes + &byteData.front();

    /* read file */
    if (setjmp(png_jmpbuf(png_ptr)))
        abort_("[read_png_file] Error during read_image");

    png_read_image(png_ptr, &rowData.front());

    fclose(fp);

    for (size_t y = 0; y < height; y++) {
        png_byte *b = rowData[y];
        for (size_t x = 0; x < width; x++) {
            auto v = pixels[y * width + x];

            v.r = (double)(*b++) / 255.f;
            v.g = (double)(*b++) / 255.f;
            v.b = (double)(*b++) / 255.f;
            if (png_get_channels(png_ptr, info_ptr) == 4) {
                b++;
            }
        }
    }
}
