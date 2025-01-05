#define BIT_N (1 << 0)
#define BIT_E (1 << 1)
#define BIT_S (1 << 2)
#define BIT_W (1 << 3)
#define BIT_BW (1 << 4)

#define BITS_NEWS 0x0f

char get(
    __global unsigned char *map,
    int x,
    int y,
    int width,
    int height) {
  if (x < 0) {
    x += width;
  } else if (x >= width) {
    x -= width;
  }
  if (y < 0) {
    y += height;
  } else if (y >= height) {
    y -= height;
  }
  return map[y * width + x];
}

/*
  - At a white square, turn 90° clockwise,
    flip the color of the square.
  - At a black square, turn 90° counterclockwise,
    flip the color of the square.
*/
__kernel void la_rotate_and_flip(
    __global unsigned char *src,
    __global unsigned char *dst,
    int width,
    int height) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if (x >= width || y >= height) return;
  const char c = get(src, x, y, width, height);
  char c_news = 0;
  char c_bw = c & BIT_BW;
  if (!c_bw) {
    // At a white square, turn 90° clockwise
    c_news =
      (((c & BIT_N) != 0) ? BIT_E : 0) |
      (((c & BIT_E) != 0) ? BIT_S : 0) |
      (((c & BIT_S) != 0) ? BIT_W : 0) |
      (((c & BIT_W) != 0) ? BIT_N : 0);
  } else {
    // At a black square, turn 90° counterclockwise
    c_news =
      (((c & BIT_N) != 0) ? BIT_W : 0) |
      (((c & BIT_E) != 0) ? BIT_N : 0) |
      (((c & BIT_S) != 0) ? BIT_E : 0) |
      (((c & BIT_W) != 0) ? BIT_S : 0);
  }
  if (c_news) {
    // flip the color of the square
    c_bw = (~c_bw) & BIT_BW;
  }
  dst[y * width + x] = c_news | c_bw;
}

/*
  Ants move forward one unit.
*/
__kernel void la_forward(
    __global unsigned char *src,
    __global unsigned char *dst,
    int width,
    int height) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if (x >= width || y >= height) return;
  const char c_n = get(src, x, y - 1, width, height);
  const char c_e = get(src, x + 1, y, width, height);
  const char c_s = get(src, x, y + 1, width, height);
  const char c_w = get(src, x - 1, y, width, height);
  const char c = get(src, x, y, width, height);
  char c_news = 0;
  if ((c_s & BIT_N) != 0) {
    c_news |= BIT_N;
  }
  if ((c_w & BIT_E) != 0) {
    c_news |= BIT_E;
  }
  if ((c_n & BIT_S) != 0) {
    c_news |= BIT_S;
  }
  if ((c_e & BIT_W) != 0) {
    c_news |= BIT_W;
  }
  const char c_bw = c & BIT_BW;
  dst[y * width + x] = c_news | c_bw;
}

/*
  Draw field image
 */
__kernel void la_draw_image(
    __global unsigned char *field,
    __write_only image2d_t image,
    int width,
    int height) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if (x >= width || y >= height) return;
  const char c = get(field, x, y, width, height);
  float r = ((c & BIT_BW) != 0) ? 0.0 : 1.0;
  float g = r;
  float b = r;
  if ((c & BITS_NEWS) != 0) {
    r = 1.0;
    g = 0.0;
    b = 0.0;
  }
  const float4 pixel = (float4)(r, g, b, 255);
  write_imagef(image, (int2)(x,y), pixel);
}
