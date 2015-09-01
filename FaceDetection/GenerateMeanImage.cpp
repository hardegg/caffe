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

int main(int argc, const char *argv[])
{
    BlobProto mean_blob;
    int nChannels = 3;
    int width = 48;
    int height = 48;
    mean_blob.set_num(1);
    mean_blob.set_channels(3);
    mean_blob.set_height(height);
    mean_blob.set_width(width);
    int size_in_datum = nChannels * width * height;
    for (int i = 0; i < size_in_datum; ++i) {
        mean_blob.add_data(128.);
    }
    WriteProtoToBinaryFile(mean_blob, "/home/fanglin/caffe/FaceDetection/models/48net_mean.binrayproto");
    
    return 0;
}
