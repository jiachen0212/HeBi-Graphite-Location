/*******************************************************************************
 * FILENAME:      example2.cpp
 *
 * AUTHORS:       Mo Yu
 *
 * START DATE:    2022.02.22 21:45:20
 *
 * CONTACT:       yu.mo@smartmore.com
 *******************************************************************************/

#include <nlohmann/json.hpp>
#include <fstream>
#include <string>
#include <chrono>
#include <smorevision/core/toolbase.hpp>
#include "location.h"

#ifndef CONFIG_DIR
#define CONFIG_DIR "/home/moyu/HeBi-Graphite-Location/data/20220302/"
#endif


using json = nlohmann::json;

int main()
{
    cv::Mat gray = cv::imread(std::string(CONFIG_DIR) + "1.png", 0);
    cv::Rect roi = cv::Rect(450, 1150, 4500, 2150);
    std::vector<cv::Rect> ignore_rois = {
            {650, 1400, 4150, 1600},
    };
    int search_num = 1;
    int down_sample_factor = 3;

    std::string config_path = std::string(CONFIG_DIR) + "1.json";
    std::ifstream file(config_path);
    json conf;
    file >> conf;
    file.close();

    std::vector<std::vector<cv::Point2f>> mask_contours;
    std::vector<std::vector<cv::Point2f>> mask_rois;
    auto masks = conf["shapes"];
    for (const auto &mask: masks)
    {
        std::vector<cv::Point2f> contours;
        auto points = mask["points"];
        auto label = mask["label"];
        for (const auto &p: points)
        {
            cv::Point2f point(p[0].get<float>(), p[1].get<float>());
            contours.emplace_back(point);
        }
        if (label == "roi")
            mask_rois.emplace_back(contours);
        else
            mask_contours.emplace_back(contours);
    }

    caliper::Params caliper_params = caliper::Params(
            6, 30, caliper::Mode::kStandard,
            caliper::Transition::kNegative,
            caliper::Select::kStrongest,
            caliper::Interpolation::kBilinear);

    findline::Params line1_params(
            shape2d::LineSegment(math::Vector2d(860, 1230),
                                 math::Vector2d(4520, 1180)),
            50,
            120,
            20,
            0,
            caliper_params);

    findline::Params line2_params(
            shape2d::LineSegment(math::Vector2d(4860, 1460),
                                 math::Vector2d(4880, 2890)),
            50,
            120,
            20,
            0,
            caliper_params);

    findline::Params line3_params(
            shape2d::LineSegment(math::Vector2d(4520, 3200),
                                 math::Vector2d(860, 3250)),
            50,
            120,
            20,
            0,
            caliper_params);

    findline::Params line4_params(
            shape2d::LineSegment(math::Vector2d(530, 2890),
                                 math::Vector2d(550, 1460)),
            50,
            120,
            20,
            0,
            caliper_params);

    Locations locator1;
    locator1.Train(gray,
                   roi,
                   ignore_rois,
                   search_num,
                   down_sample_factor,
                   line1_params,
                   line2_params,
                   line3_params,
                   line4_params,
                   mask_contours,
                   mask_rois);

    std::ifstream image_list_file(std::string(CONFIG_DIR) + "2.txt");
    std::string image_path;
    while(std::getline(image_list_file, image_path))
    {
        std::cout << "path: " << image_path << std::endl;
        cv::Mat img = cv::imread(image_path);
        cv::Mat render, unit_mask = cv::Mat::zeros(img.size(), CV_8U);
        std::vector<std::vector<cv::Point2f>> unit_contours;
        auto start = std::chrono::system_clock::now();
        locator1.Inference(img, render, unit_contours);
        auto end = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "cost_time: " << duration.count() << "ms" << std::endl;
        cv::imwrite(image_path + "_result.png", render);

        int i = 1;
        for (const auto& contours: unit_contours)
        {
            std::vector<cv::Point> ps;
            for (const auto &p: contours)
            {
                ps.emplace_back(cv::Point((int)p.x, (int)p.y));
            }
            std::vector<std::vector<cv::Point>> points = {ps};
            cv::drawContours(unit_mask, points, 0, i++, -1);
        }
        cv::imwrite(image_path + "_unit_mask.png", unit_mask);
    }
    image_list_file.close();

    return 0;
}