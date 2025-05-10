#include "FS.h"
#include "SPIFFS.h"
#include <Arduino.h>
#include <U8g2lib.h>
#include <freertos/FreeRTOS.h>
#include <freertos/queue.h>

#ifdef U8X8_HAVE_HW_SPI
#include <SPI.h>
#endif
#ifdef U8X8_HAVE_HW_I2C
#include <Wire.h>
#endif

#define FORMAT_SPIFFS_IF_FAILED true
#define VIDEO_FILE_PATH "/video.data" // Video file path

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define FRAME_SIZE (SCREEN_WIDTH * SCREEN_HEIGHT / 8) // Each frame is 1 KB
#define QUEUE_SIZE 100 // Queue size, 100 frames, ~100 KB

#define OLED_SCL 16 // RX2 = 16
#define OLED_SDA 17 // TX2 = 17

// Task parameters structure
struct TaskParams {
    fs::FS &fs;
    const char *path;
};

U8G2_SSD1306_128X64_NONAME_F_HW_I2C u8g2(U8G2_R0, U8X8_PIN_NONE, OLED_SCL, OLED_SDA);

QueueHandle_t frameQueue; // Queue handle for frame data
uint32_t fps = 0; // Frame rate 

void displayCenteredText(const char *text)
{
    u8g2.clearBuffer();
    int text_width = u8g2.getStrWidth(text);
    int x = (u8g2.getDisplayWidth() - text_width) / 2;
    int y = (u8g2.getDisplayHeight() + u8g2.getAscent()) / 2;
    u8g2.drawStr(x, y, text);
    u8g2.sendBuffer();
}

bool readFrameRate(fs::FS &fs, const char *path, uint32_t &fps)
{
    File file = fs.open(path);
    if (!file || file.isDirectory()) {
        return false;
    }
    file.read((uint8_t *)&fps, sizeof(fps));
    file.close();
    return true;
}

void readFramesTask(void *parameter)
{
    TaskParams *params = (TaskParams *)parameter;
    
    File file = params->fs.open(params->path);
    if (!file || file.isDirectory()) {
        Serial.printf("Failed to open %s!\n", params->path);
        displayCenteredText("Failed to open file!");
        delay(2000);
        vTaskDelete(NULL);
        return;
    }

    // Skip frame rate (already read in setup)
    file.seek(sizeof(uint32_t));

    // Allocate buffer for a single frame
    unsigned char *frame = (unsigned char *)malloc(FRAME_SIZE);
    if (!frame) {
        Serial.println("Failed to allocate frame buffer!");
        displayCenteredText("Memory allocation failed!");
        delay(2000);
        file.close();
        vTaskDelete(NULL);
        return;
    }

    while (file.available() >= FRAME_SIZE) {
        // Read one frame
        for (int i = 0; i < FRAME_SIZE; i++) {
            frame[i] = file.read();
        }

        // Wait for queue space and send frame
        if (xQueueSend(frameQueue, frame, portMAX_DELAY) != pdTRUE) {
            Serial.println("Failed to send frame to queue!");
            displayCenteredText("Failed to send frame to queue!");
            delay(2000);
            break;
        }
    }

    // Close file and free memory
    file.close();
    free(frame);
    Serial.println("Frame reading complete.");
    vTaskDelete(NULL);
}

void playVideo(fs::FS &fs, const char *path, uint32_t fps)
{
    // Create queue for frames
    frameQueue = xQueueCreate(QUEUE_SIZE, FRAME_SIZE);
    if (!frameQueue) {
        Serial.println("Failed to create queue!");
        displayCenteredText("Queue creation failed!");
        delay(2000);
        return;
    }

    // Prepare parameters for read task
    TaskParams taskParams = {fs, path};

    // Allocate buffer for display
    unsigned char *frame = (unsigned char *)malloc(FRAME_SIZE);
    if (!frame) {
        Serial.println("Failed to allocate display buffer!");
        displayCenteredText("Memory allocation failed!");
        delay(2000);
        vQueueDelete(frameQueue);
        return;
    }

    // Start asynchronous frame reading task
    // xTaskCreate(readFramesTask, "ReadFramesTask", 4096, (void *)&taskParams, 1, NULL);
    xTaskCreatePinnedToCore(readFramesTask, "ReadFramesTask", 4096, (void *)&taskParams, 1, NULL, 0);

    // Wait for the first frame to be available
    xQueuePeek(frameQueue, frame, portMAX_DELAY);

    // Calculate delay based on frame rate
    uint32_t delay_us = 1000000 / fps;

    Serial.println("Playing...");

    // Retrieve frames from queue and display
    while (xQueueReceive(frameQueue, frame, 0) == pdTRUE) {
        u8g2.clearBuffer();
        u8g2.drawXBM(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, frame);
        u8g2.sendBuffer();
        delayMicroseconds(delay_us);
    }

    // Clean up
    free(frame);
    vQueueDelete(frameQueue);
    Serial.println("Playback complete.");
}

void setup()
{
    Serial.begin(115200);

    u8g2.setBusClock(400000);
    if (!u8g2.begin()) {
        Serial.println("OLED init failed!");
        displayCenteredText("OLED init failed!");
        delay(2000);
        while (true) {
            delay(1000);
        }
    }
    u8g2.setFont(u8g2_font_ncenB08_tr);

    displayCenteredText("OLED Video Player");
    delay(2000);

    if (!SPIFFS.begin(FORMAT_SPIFFS_IF_FAILED)) {
        Serial.println("SPIFFS Mount Failed!");
        displayCenteredText("SPIFFS Mount Failed!");
        delay(2000);
        while (true) {
            delay(1000);
        }
    }

    // Read frame rate
    if (!readFrameRate(SPIFFS, VIDEO_FILE_PATH, fps)) {
        Serial.println("Failed to read frame rate!");
        displayCenteredText("Failed to read fps!");
        delay(2000);
        while (true) {
            delay(1000);
        }
    }
    Serial.printf("Frame rate: %d.\n", fps);
    char fps_text[32];
    sprintf(fps_text, "Frame Rate: %d", fps);
    displayCenteredText(fps_text);
    delay(2000);
}

void loop()
{
    playVideo(SPIFFS, VIDEO_FILE_PATH, fps);
}
