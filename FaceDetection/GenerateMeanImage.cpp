#include <stdint.h>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

using std::max;
using std::pair;
using boost::scoped_ptr;

void GenerateConstantMean(int width, int height, int nChannels, float value, const string& filepath)
{
    BlobProto mean_blob;
    mean_blob.set_num(1);
    mean_blob.set_channels(nChannels);
    mean_blob.set_height(height);
    mean_blob.set_width(width);
    int size_in_datum = nChannels * width * height;
    for (int i = 0; i < size_in_datum; ++i) {
        mean_blob.add_data(128.);
    }
    WriteProtoToBinaryFile(mean_blob, filepath); 
}

int main(int argc, const char *argv[])
{
    string filepath = "/home/fanglin/caffe/FaceDetection/models/deploy/12net_mean_const128.binaryproto";
   
       
    return 0;
}
