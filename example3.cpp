/*******************************************************************************
 * FILENAME:      example3.cpp
 *
 * AUTHORS:       Mo Yu
 *
 * START DATE:    2022.02.22 23:14:11
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
#define CONFIG_DIR "/home/moyu/HeBi-Graphite-Location/data/20220309/"
#endif


using json = nlohmann::json;

int main()
{
    cv::Mat gray = cv::imread(std::string(CONFIG_DIR) + "3.png", 0);
    cv::Rect roi = cv::Rect(700, 800, 3000, 6300);
    std::vector<cv::Rect> ignore_rois = {
            {1000, 1150, 2500, 5600},
    };
    int search_num = 2;
    int down_sample_factor = 8;

    std::string config_path = std::string(CONFIG_DIR) + "3.json";
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
            2, 30, caliper::Mode::kStandard,
            caliper::Transition::kNegative,
            caliper::Select::kLast,
            caliper::Interpolation::kBilinear);

    findline::Params line1_params(
            shape2d::LineSegment(math::Vector2d(1200, 950),
                                 math::Vector2d(3140, 950)),
            50,
            100,
            20,
            0,
            caliper_params);

    findline::Params line2_params(
            shape2d::LineSegment(math::Vector2d(3600, 1400),
                                 math::Vector2d(3600, 6500)),
            50,
            100,
            20,
            0,
            caliper_params);

    findline::Params line3_params(
            shape2d::LineSegment(math::Vector2d(3140, 6930),
                                 math::Vector2d(1200, 6930)),
            50,
            100,
            20,
            0,
            caliper_params);

    findline::Params line4_params(
            shape2d::LineSegment(math::Vector2d(850, 6500),
                                 math::Vector2d(850, 1400)),
            50,
            100,
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

    std::ifstream image_list_file(std::string(CONFIG_DIR) + "3.txt");
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
        if (!render.empty())
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