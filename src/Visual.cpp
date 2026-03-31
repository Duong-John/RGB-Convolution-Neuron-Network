#include "Visual.hpp"

Visual::Visual(int w, int h) : window_width(w), window_height(h) {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        throw std::runtime_error("SDL could not initialize!");
    }
    window = SDL_CreateWindow("CNN Feature Maps Visualization", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, window_width, window_height, SDL_WINDOW_SHOWN);
    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
}

Visual::~Visual() {
    if (renderer) SDL_DestroyRenderer(renderer);
    if (window) SDL_DestroyWindow(window);
    SDL_Quit();
}

void Visual::normalize_and_draw_channel(float* channel_data, int H, int W, int draw_x, int draw_y, int cell_size) {

    float min_val = 1e9f, max_val = -1e9f;
    for (int i = 0; i < H * W; ++i) {
        if (channel_data[i] < min_val) min_val = channel_data[i];
        if (channel_data[i] > max_val) max_val = channel_data[i];
    }
    float range = max_val - min_val;
    if (range < 1e-6f) range = 1.0f; 


    float pixel_w = (float)cell_size / W;
    float pixel_h = (float)cell_size / H;


    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            float val = channel_data[h * W + w];
           
            Uint8 color = static_cast<Uint8>(((val - min_val) / range) * 255.0f);

            SDL_Rect pixel_rect;
            pixel_rect.x = draw_x + static_cast<int>(w * pixel_w);
            pixel_rect.y = draw_y + static_cast<int>(h * pixel_h);
            pixel_rect.w = static_cast<int>(std::ceil(pixel_w));
            pixel_rect.h = static_cast<int>(std::ceil(pixel_h));

            SDL_SetRenderDrawColor(renderer, color, color, color, 255);
            SDL_RenderFillRect(renderer, &pixel_rect);
        }
    }
}

void Visual::draw_feature_maps(float* h_data, int C, int H, int W, const std::string& layer_name) {
    SDL_SetRenderDrawColor(renderer, 30, 30, 30, 255);
    SDL_RenderClear(renderer);

    int cols = std::ceil(std::sqrt(C));
    int rows = std::ceil((float)C / cols);

    int padding = 5;
    int cell_w = (window_width - padding * (cols + 1)) / cols;
    int cell_h = (window_height - 50 - padding * (rows + 1)) / rows;
    int cell_size = std::min(cell_w, cell_h);

    int start_x = (window_width - (cols * cell_size + (cols - 1) * padding)) / 2;
    int start_y = 30;

    std::cout << "\n\033[1;36m[Visual] Drawing " << C << " Feature Maps for " << layer_name << "...\033[0m\n";

    for (int c = 0; c < C; ++c) {
        int row = c / cols;
        int col = c % cols;

        int draw_x = start_x + col * (cell_size + padding);
        int draw_y = start_y + row * (cell_size + padding);

        
        float* channel_data = h_data + c * (H * W);
        
        normalize_and_draw_channel(channel_data, H, W, draw_x, draw_y, cell_size);
    }

    SDL_RenderPresent(renderer);
}

void Visual::wait_for_close() {
    std::cout << "[Visual] Window is open. Please close the window (click 'X') to continue...\n";
    SDL_Event e;
    bool quit = false;
    while (!quit) {
        while (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_QUIT) {
                quit = true;
            }
            if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE) {
                quit = true;
            }
        }
        SDL_Delay(50);
    }
}