#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <dirent.h>
#include <vector>
#include <iostream>
#include <map>
#include <chrono>
#include "headers/json.hpp"
#include "headers/hog.hpp"


using namespace std;
using namespace cv;
using namespace ml;
using json = nlohmann::json;


int bin=9;
int patch_size=8;

int treeSize=100;
int treeDepth=16;//16 default
int max_cat=15;//15 default
int min_sample_count=10;//10 default
int nactiveVar=4;//4 default
//PCA parameters
int var=100;

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

inline TermCriteria TC(int iters, double eps) {
    return TermCriteria(TermCriteria::MAX_ITER + (eps > 0 ? TermCriteria::EPS : 0), iters, eps);
}

static Ptr<TrainData> prepareTrainData(const Mat& data, const Mat& responses, long int ntrain_samples) {
    Mat sample_idx = Mat::zeros( 1, data.rows, CV_8U );
    cout<<"Samples idx size: "<<sample_idx.size()<<endl;
    Mat train_samples = sample_idx.colRange(0, ntrain_samples);
    train_samples.setTo(Scalar::all(1));

    int nvars = data.cols;
    Mat var_type( nvars + 1, 1, CV_8U );
    var_type.setTo(Scalar::all(VAR_ORDERED));
    var_type.at<uchar>(nvars) = VAR_CATEGORICAL;

    return TrainData::create(data, ROW_SAMPLE, responses,
                             noArray(), sample_idx, noArray(), var_type);
}


void trainRtree(Mat& samples,Mat& labelsMat){
  //double nsamples_all = samples.rows;
  //double ntrain_samples = (double)(nsamples_all*1.0);
  //cout<<"Number of samples: "<<samples.rows;
  Ptr<TrainData> tdata = prepare_train_data(samples, labelsMat, samples.rows);
  Ptr<RTrees> model;
  model = RTrees::create();
  model->setMaxDepth(treeDepth);
  model->setMinSampleCount(20);//10 default - 20 seems best;
  model->setRegressionAccuracy(0);
  model->setUseSurrogates(false);
  model->setMaxCategories(max_cat);
  model->setPriors(Mat());
  model->setCalculateVarImportance(true);//true default
  model->setActiveVarCount(nactiveVar);
  model->setTermCriteria(TC(treeSize,0.01f));
  model->train(tdata);
  cout << "Number of trees: " << model->getRoots().size() << endl;
  model->save("trees.yml");
}

int main(){
  vector<Mat> images;
  vector<string> paths;
  string dir="images/";
  listFiles(dir,paths);
  json out,out2;
  Mat labelsMat(paths.size(),1, CV_32S);
  int label=-1;
  sort(paths.begin(),paths.end());
  cout<<"Total number of images: "<<paths.size();
  hog h(bin,patch_size);
  h.info();
  Mat samples;
  for(int i=0;i<paths.size();i++){
    Mat grad;
    vector<string> splits=split(paths[i],"/");
    string file=split(splits[2],".")[0];
    string folder=splits[1];
    if(file==folder){
      cout<<"Extracting HOGs in "<<file<<endl;
      label++;
      out[to_string(label)]=paths[i];
      out2[to_string(label)]=file;
    }
    Mat temp=imread(paths[i],IMREAD_GRAYSCALE);
    grad=h.compute(temp);
    grad=grad.reshape(1,1);
    samples.push_back(grad);
    grad.release();
    temp.release();
    labelsMat.at<int>(i)=label;
  }
  ofstream file("labels_debug.json");
  file << setw(4) << out << endl;
  ofstream file2("labels.json");
  file2 << setw(4) << out2 << endl;

  cout<<"Created labels"<<endl;

  cout<<"Training random forest classifier..."<<endl;
  cout<<"Number of samples: "<<samples.rows<<endl;

  Ptr<TrainData> tdata = prepareTrainData(samples, labelsMat, samples.rows);
  cout<<"prepped training data!!"<<endl;
  Ptr<RTrees> model;
  //long int minS=samples.rows*0.01;
  model = RTrees::create();
  model->setMaxDepth(treeDepth);
  model->setMinSampleCount(20);//10 default - 20 seems best;
  model->setMaxCategories(max_cat);
  model->setPriors(Mat());
  model->setActiveVarCount(nactiveVar);
  model->setCalculateVarImportance(true);//true default

  model->setTermCriteria(TC(treeSize,0.01f));
  model->train(tdata);
  //model->train(samples,ROW_SAMPLE,labelsMat)
  cout << "Number of trees: " << model->getRoots().size() << endl;
  model->save("trees.yml");
  //samples.release();
  //labelsMat.release();
}
