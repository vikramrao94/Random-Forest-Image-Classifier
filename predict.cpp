#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ml.hpp>
#include <dirent.h>
#include <vector>
#include <iostream>
#include <map>
#include <chrono>
#include "nlohmann/json.hpp"
#include "hog.hpp"


using namespace std;
using namespace cv;
using namespace xfeatures2d;
using namespace ml;
using json = nlohmann::json;

int num_comp=100;


auto startTime(){
  return(chrono::steady_clock::now());
}

float timeElapsed(auto start){
  auto end = chrono::steady_clock::now();
  auto diff = end - start;
  return(chrono::duration <double, milli> (diff).count());
}

Mat createOne(vector<Mat> & images, int cols, int min_gap_size)
{
    // let's first find out the maximum dimensions
    int max_width = 0;
    int max_height = 0;
    for ( int i = 0; i < images.size(); i++) {
        // check if type is correct
        // you could actually remove that check and convert the image
        // in question to a specific type
        if ( i > 0 && images[i].type() != images[i-1].type() ) {
            cerr << "WARNING:createOne failed, different types of images";
            return Mat();
        }
        max_height = max(max_height, images[i].rows);
        max_width = max(max_width, images[i].cols);
    }
    // number of images in y direction
    int rows = ceil(images.size() / cols);

    // create our result-matrix
    Mat result = Mat::zeros(rows*max_height + (rows-1)*min_gap_size,
                                    cols*max_width + (cols-1)*min_gap_size, images[0].type());
    size_t i = 0;
    int current_height = 0;
    int current_width = 0;
    for ( int y = 0; y < rows; y++ ) {
        for ( int x = 0; x < cols; x++ ) {
            if ( i >= images.size() ) // shouldn't happen, but let's be safe
                return result;
            // get the ROI in our result-image
            Mat to(result,
                       Range(current_height, current_height + images[i].rows),
                       Range(current_width, current_width + images[i].cols));
            // copy the current image to the ROI
            images[i++].copyTo(to);
            current_width += max_width + min_gap_size;
        }
        // next line - reset width and update height
        current_width = 0;
        current_height += max_height + min_gap_size;
    }
    return result;
}

void listFiles(const string &path, vector<string> &cb) {
    if (auto dir = opendir(path.c_str())) {
        while (auto f = readdir(dir)) {
            if (!f->d_name || f->d_name[0] == '.') continue;
            if (f->d_type == DT_DIR)
                listFiles(path + f->d_name + "/", cb);

            if (f->d_type == DT_REG)
                cb.push_back(path + f->d_name);
        }
        closedir(dir);
    }
}

vector<string> split(string str,string sep){
    char* cstr=const_cast<char*>(str.c_str());
    char* current;
    vector<string> arr;
    current=strtok(cstr,sep.c_str());
    while(current!=NULL){
        arr.push_back(current);
        current=strtok(NULL,sep.c_str());
    }
    return arr;
}

int main(){
  vector<string> paths;
  string dir="Everything/";
  listFiles(dir,paths);
  Ptr<RTrees> model=Algorithm::load<RTrees>("trees.yml");
  //Ptr<SVM> model=Algorithm::load<SVM>("svm_hog.yml");

  Mat gradient;
  ifstream ifs("labels_debug.json");
  json j = json::parse(ifs);
  hog h(9,8);
  vector<int> stats;
  for(int i=0;i<paths.size();i++){
    Mat img=imread(paths[i],IMREAD_GRAYSCALE);
    resize(img, img, Size(200,300), CV_INTER_LINEAR);
    Rect roi(20,50,160,160);
    Mat region=img(roi);
    //medianBlur ( region, region, 3 );
    auto start=startTime();
    Size pos_size=img.size();
    //pos_size = pos_size / 8 * 8;
    //computeHOG(pos_size,img,gradient);
    gradient=h.compute(region);
    //gradient=gradient.reshape(1,1);
    //PCA pca(gradient, Mat(), PCA::DATA_AS_ROW,num_comp);
    //Mat reduced = pca.project(gradient);
    //int res=model->predict(gradient);
    Mat votes;
    model->getVotes(gradient,votes,0);
    //cout<<"votes: "<<votes.type()<<endl;
    int max=0;
    int res;
    for (int i=0;i<votes.cols;i++){
      int v=votes.at<int>(1,i);
      if(v>max){
        res=votes.at<int>(0,i);
        max=v;
      }
    }
    //cout<<"Result from voting: "<<r<<" and its votes: "<<max<<endl;
    //int res=(int)model->predict(gradient);
    stats.push_back(max);
    string timeTaken=" Time: "+to_string(timeElapsed(start))+" ms";
    cout<<"result: "<<res<<timeTaken<<endl;
    Mat match=imread(j.at(to_string(res)),IMREAD_GRAYSCALE);
    vector<Mat> temp;
    temp.push_back(img);
    temp.push_back(match);
    Mat combined=createOne(temp, 1,5);
    putText(combined,
            timeTaken,
            Point(20,50), // Coordinates
            FONT_HERSHEY_COMPLEX_SMALL, // Font
            0.5, // Scale. 2.0 = 2x bigger
            Scalar(255,255,255), // BGR Color
            1, // Line Thickness (Optional)
            CV_AA);
    imwrite("out/"+to_string(i)+".png",combined);
  }
  sort(stats.begin(),stats.end());
  auto mean = accumulate(begin(stats),end(stats), 0.0) / stats.size();
  cout<<"Statistics.."<<endl;
  cout<<"Min: "<<stats[0]<<" Max: "<<stats[stats.size()-1]<<endl;
  cout<<"Mean: "<<mean<<endl;
  return 0;
}
