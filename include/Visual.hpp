#ifndef VISUAL_HPP
#define VISUAL_HPP

#include "Header.h"

class Visual {
private:
    int window_width;
    int window_height;
    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;

    void normalize_and_draw_channel(float* channel_data, int H, int W, int draw_x, int draw_y, int cell_size);

public:
    Visual(int w = 1200, int h = 800);
    ~Visual();

    
    void draw_feature_maps(float* h_data, int C, int H, int W, const std::string& layer_name);
    
    void wait_for_close();
};

#endif