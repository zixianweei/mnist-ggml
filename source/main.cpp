#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>

#include <ggml/ggml.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#define MNIST_IW_IH 28
#define MNIST_IWPIH 784

///////////////////////////////////////////////////////////////////////////////
// LOG_*

#define LOG_INFO(format, ...) fprintf(stdout, format, ##__VA_ARGS__)
#define LOG_WARN(format, ...) fprintf(stderr, format, ##__VA_ARGS__)

///////////////////////////////////////////////////////////////////////////////
// mnist ggml

struct mnist_t {
  struct ggml_context* ctx{nullptr};

  // conv1
  struct ggml_tensor* conv1_weight{nullptr};  // 32 1 3 3
  struct ggml_tensor* conv1_bias{nullptr};    // 1 32 26 26

  // conv2
  struct ggml_tensor* conv2_weight{nullptr};  // 64 32 3 3
  struct ggml_tensor* conv2_bias{nullptr};    // 1 64 24 24

  // fc1
  struct ggml_tensor* fc1_weight{nullptr};  // 128 9216
  struct ggml_tensor* fc1_bias{nullptr};    // 128

  // fc2
  struct ggml_tensor* fc2_weight{nullptr};  // 10 128
  struct ggml_tensor* fc2_bias{nullptr};    // 10
};

bool load_model(const char* filename, mnist_t& model) {
  LOG_INFO("%s: loading model from '%s'\n", __func__, filename);

  auto fin = std::ifstream(filename, std::ios::binary);
  if (fin.fail()) {
    LOG_WARN("%s: failed to open '%s'\n", __func__, filename);
    return false;
  }

  // verify magic
  {
    uint32_t magic;
    fin.read((char*)&magic, sizeof(magic));
    if (magic != GGML_FILE_MAGIC) {
      LOG_WARN("%s: invalid model file '%s' (bad magic)\n", __func__, filename);
      return false;
    }
  }

  // create the ggml context
  {
    struct ggml_init_params params = {
        1024 * 1024 * 8,
        nullptr,
        false,
    };

    model.ctx = ggml_init(params);
    if (!model.ctx) {
      LOG_WARN("%s: ggml_init() failed\n", __func__);
      return false;
    }
  }
  auto& ctx = model.ctx;

  // conv1_weight
  {
    int32_t n_dims;
    fin.read(reinterpret_cast<char*>(&n_dims), sizeof(n_dims));
    {
      int32_t ne[4] = {1, 1, 1, 1};
      for (int i = 0; i < n_dims; i++) {
        fin.read(reinterpret_cast<char*>(&ne[i]), sizeof(ne[i]));
      }
      model.conv1_weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, ne[0], ne[1], ne[2], ne[3]);
      fin.read(reinterpret_cast<char*>(model.conv1_weight->data), ggml_nbytes(model.conv1_weight));
      ggml_set_name(model.conv1_weight, "conv1_weight");
    }
  }

  // conv1_bias
  {
    int32_t n_dims;
    fin.read(reinterpret_cast<char*>(&n_dims), sizeof(n_dims));
    {
      int32_t ne[4] = {1, 1, 1, 1};
      for (int i = 0; i < n_dims; i++) {
        fin.read(reinterpret_cast<char*>(&ne[i]), sizeof(ne[i]));
      }
      model.conv1_bias = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);
      fin.read(reinterpret_cast<char*>(model.conv1_bias->data), ggml_nbytes(model.conv1_bias));
      ggml_set_name(model.conv1_bias, "conv1_bias");
    }
  }

  // conv2_weight
  {
    int32_t n_dims;
    fin.read(reinterpret_cast<char*>(&n_dims), sizeof(n_dims));
    {
      int32_t ne[4] = {1, 1, 1, 1};
      for (int i = 0; i < n_dims; i++) {
        fin.read(reinterpret_cast<char*>(&ne[i]), sizeof(ne[i]));
      }
      model.conv2_weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, ne[0], ne[1], ne[2], ne[3]);
      fin.read(reinterpret_cast<char*>(model.conv2_weight->data), ggml_nbytes(model.conv2_weight));
      ggml_set_name(model.conv2_weight, "conv2_weight");
    }
  }

  // conv2_bias
  {
    int32_t n_dims;
    fin.read(reinterpret_cast<char*>(&n_dims), sizeof(n_dims));
    {
      int32_t ne[4] = {1, 1, 1, 1};
      for (int i = 0; i < n_dims; i++) {
        fin.read(reinterpret_cast<char*>(&ne[i]), sizeof(ne[i]));
      }
      model.conv2_bias = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);
      fin.read(reinterpret_cast<char*>(model.conv2_bias->data), ggml_nbytes(model.conv2_bias));
      ggml_set_name(model.conv2_bias, "conv2_bias");
    }
  }

  // fc1_weight
  {
    int32_t n_dims;
    fin.read(reinterpret_cast<char*>(&n_dims), sizeof(n_dims));
    {
      int32_t ne[4] = {1, 1, 1, 1};
      for (int i = 0; i < n_dims; i++) {
        fin.read(reinterpret_cast<char*>(&ne[i]), sizeof(ne[i]));
      }
      model.fc1_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne[0], ne[1]);
      fin.read(reinterpret_cast<char*>(model.fc1_weight->data), ggml_nbytes(model.fc1_weight));
      ggml_set_name(model.fc1_weight, "fc1_weight");
    }
  }

  // fc1_bias
  {
    int32_t n_dims;
    fin.read(reinterpret_cast<char*>(&n_dims), sizeof(n_dims));
    {
      int32_t ne[4] = {1, 1, 1, 1};
      for (int i = 0; i < n_dims; i++) {
        fin.read(reinterpret_cast<char*>(&ne[i]), sizeof(ne[i]));
      }
      model.fc1_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne[0]);
      fin.read(reinterpret_cast<char*>(model.fc1_bias->data), ggml_nbytes(model.fc1_bias));
      ggml_set_name(model.fc1_bias, "fc1_bias");
    }
  }

  // fc2_weight
  {
    int32_t n_dims;
    fin.read(reinterpret_cast<char*>(&n_dims), sizeof(n_dims));
    {
      int32_t ne[4] = {1, 1, 1, 1};
      for (int i = 0; i < n_dims; i++) {
        fin.read(reinterpret_cast<char*>(&ne[i]), sizeof(ne[i]));
      }
      model.fc2_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne[0], ne[1]);
      fin.read(reinterpret_cast<char*>(model.fc2_weight->data), ggml_nbytes(model.fc2_weight));
      ggml_set_name(model.fc2_weight, "fc2_weight");
    }
  }

  // fc2_bias
  {
    int32_t n_dims;
    fin.read(reinterpret_cast<char*>(&n_dims), sizeof(n_dims));
    {
      int32_t ne[4] = {1, 1, 1, 1};
      for (int i = 0; i < n_dims; i++) {
        fin.read(reinterpret_cast<char*>(&ne[i]), sizeof(ne[i]));
      }
      model.fc2_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne[0]);
      fin.read(reinterpret_cast<char*>(model.fc2_bias->data), ggml_nbytes(model.fc2_bias));
      ggml_set_name(model.fc2_bias, "fc2_bias");
    }
  }

  return true;
}

int inference(mnist_t& model, const float* image, const char* graphname) {
  // 1. init context for inference;
  size_t buf_size = 1024 * 1024 * 8;
  void* buf = malloc(buf_size);  // managed by ggml memory pool
  struct ggml_init_params params {
    buf_size, buf, false,
  };
  ggml_context* ctx = ggml_init(params);

  ggml_tensor* x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, MNIST_IW_IH, MNIST_IW_IH, 1, 1);
  memcpy(x->data, image, ggml_nbytes(x));
  ggml_set_name(x, "x");

  struct ggml_cgraph* gf = ggml_new_graph(ctx);

  // 2. inferring
  // x = self.conv1(x)
  // x = F.relu(x)
  x = ggml_conv_2d(ctx, model.conv1_weight, x, 1, 1, 0, 0, 1, 1);
  x = ggml_add(ctx, model.conv1_bias, x);
  x = ggml_relu(ctx, x);
  // x = self.conv2(x)
  // x = F.relu(x)
  x = ggml_conv_2d(ctx, model.conv2_weight, x, 1, 1, 0, 0, 1, 1);
  x = ggml_add(ctx, model.conv2_bias, x);
  x = ggml_relu(ctx, x);
  // x = F.max_pool2d(x)
  x = ggml_pool_2d(ctx, x, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
  // x = torch.flatten(x, 1)
  x = ggml_reshape_2d(ctx, x, x->ne[0] * x->ne[1] * x->ne[2] * x->ne[3], 1);
  // x = self.fc1(x)
  // x = F.relu(x)
  x = ggml_mul_mat(ctx, model.fc1_weight, x);
  x = ggml_add(ctx, model.fc1_bias, x);
  x = ggml_relu(ctx, x);
  // x = self.fc2(x)
  // x = F.log_softmax(x, dim=1)
  x = ggml_mul_mat(ctx, model.fc2_weight, x);
  x = ggml_add(ctx, model.fc2_bias, x);
  x = ggml_soft_max(ctx, x);
  // TODO(zixianweei): softmax is not log_softmax.
  x = ggml_log(ctx, x);

  // 3. graph inference
  ggml_build_forward_expand(gf, x);
  ggml_graph_compute_with_ctx(ctx, gf, 1);

  ggml_graph_dump_dot(gf, nullptr, "mnist-cnn.dot");

  if (graphname != nullptr) {
    // export the compute graph for later use
    // see the "mnist-cpu" example
    ggml_graph_export(gf, graphname);

    LOG_INFO("%s: exported compute graph to '%s'\n", __func__, graphname);
  }

  // 4. postprocessing, find max probability
  float* data = ggml_get_data_f32(x);
  ggml_free(ctx);

  return std::max_element(data, data + 10) - data;
}

///////////////////////////////////////////////////////////////////////////////
// image_t

struct image_t final {
  unsigned char* data{nullptr};
  int w{};
  int h{};
  int c{};
};

void create_mnist_dataset(image_t& image) {
  image.data = (unsigned char*)malloc(MNIST_IWPIH * sizeof(unsigned char));
  image.h = MNIST_IW_IH;
  image.w = MNIST_IW_IH;
  image.c = 1;
}

///////////////////////////////////////////////////////////////////////////////
// load_image

enum {
  LOAD_image,
  LOAD_dataset,
};

bool load_image(image_t& image, const char* filename, int mode) {
  if (mode == LOAD_image) {
    image.data = stbi_load(filename, &image.w, &image.h, &image.c, STBI_grey);
    if (image.data == nullptr) {
      LOG_WARN("%s: failed to open '%s'\n", __func__, filename);
      return false;
    }
  } else if (mode == LOAD_dataset) {
    std::ifstream fin(filename, std::ios::binary);
    if (fin.fail()) {
      LOG_WARN("%s: failed to open '%s'\n", __func__, filename);
      return false;
    }
    fin.seekg(16 + MNIST_IWPIH * (rand() % 10000));
    fin.read((char*)image.data, MNIST_IWPIH * sizeof(unsigned char));
  } else {
    LOG_WARN("%s: unknown image loading mode '%d'\n", __func__, mode);
    return false;
  }
  return true;
}

void print_image(image_t& image) {
  for (int hh = 0; hh < image.h; hh++) {
    for (int ww = 0; ww < image.w; ww++) {
      // LOG_INFO("%d ", image.data[hh * image.w + ww]);
      LOG_INFO("%c ", image.data[hh * image.w + ww] > 128 ? ' ' : '*');
    }
    LOG_INFO("\n");
  }
  LOG_INFO("\n");
}

void transform_image(image_t& image, float*& image_f) {
  static float mean = 0.1307;
  static float stddev = 0.3081;
  for (int hh = 0; hh < image.h; hh++) {
    for (int ww = 0; ww < image.w; ww++) {
      image_f[hh * image.w + ww] = ((float)image.data[hh * image.w + ww] / 255.F - mean) / stddev;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// main

int main() {
  LOG_WARN("work directory: [%s]\n", std::filesystem::current_path().c_str());
  srand(time(nullptr));

  /// 1. read image (mnist test data)
  image_t image;
  create_mnist_dataset(image);
  const char* filename = "../data/MNIST/raw/t10k-images-idx3-ubyte";
  if (!load_image(image, filename, LOAD_dataset)) {
    LOG_WARN("%s: failed to load_image\n", filename);
  }
  // /// 1. read image (image)
  // image_t image;
  // const char* filename = "../data/7.png";
  // if (!load_image(image, filename, LOAD_image)) {
  //   LOG_WARN("%s: failed to load_image\n", filename);
  // }

  /// 2. print image
  print_image(image);

  /// 3. transform image to float
  float* image_f = new float[MNIST_IWPIH];
  transform_image(image, image_f);

  /// 4. load mnist model ggml
  mnist_t model;
  const char* modelname = "../data/mnist_ggml.bin";
  if (!load_model(modelname, model)) {
    LOG_WARN("%s: failed to open '%s'\n", __func__, modelname);
    return 1;
  }

  // 5. inference
  int p = inference(model, image_f, nullptr);
  LOG_INFO("MNIST inference: %d\n", p);

  // 6. cleanup
  ggml_free(model.ctx);
  stbi_image_free(image.data);  // finally, call free
  delete[] image_f;

  return 0;
}
