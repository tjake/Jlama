#ifndef DOT_H
#define DOT_H

float dot_product(const short* a, int aoffset, const short* b, int boffset, int length);

void accumulate(const short* a, const short* b, int length);

void saxpy(float a, const short* x, int xoffset,  const short* y, int yoffset, int length);

void scale(float factor, const short* t, int offset, int length);

void debug(const short* x, int length);


#endif