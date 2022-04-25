#include <iostream>
#include <arrayfire.h>
#include <eigen3/Eigen/Eigen>

#include "flashlight/fl/flashlight.h"


#include "aux.h"


typedef std::vector<RowVector*> data;



class agent {

public:
    agent();
    
    ~agent();

    void initialize();

    void train(int epochs);

    void populate(std::string filename);

    af::array af_from_eigen(Eigen::RowVectorXf* v);

    std::shared_ptr<fl::Sequential> model_ = std::make_shared<fl::Sequential>();

    //fl::MeanSquaredError loss = fl::MeanSquaredError();

    std::pair<double, double> eval_loop(fl::Sequential& model, fl::BatchDataset& dataset);

    af::array train_x, train_y;

    float learning_rate_ = 0.001;
    int batch_size_ = 64;

    int INPUT_IDX = 0;
    int TARGET_IDX = 1;

    //fl::Sequential * model_ = nullptr;
};

agent::agent(){}

agent::~agent(){}

void agent::initialize(){

    model_ = std::make_shared<fl::Sequential>();

    model_->add(fl::Linear(2,3));
    model_->add(fl::Linear(3,1));
}

void agent::train(int epochs){

    initialize();

    std::cout << " b b " << std::endl;
    
    fl::TensorDataset data({train_x, train_y});

    std::cout << " b b " << std::endl;
    
    int VAL_SIZE = 1000;
    int TRAIN_SIZE = 10000;
    int TEST_SIZE = 1000;

    std::cout << " b b " << train_x.dims() << " / " << train_y.dims() << std::endl;
    

    // Hold out a dev set
    //auto val_x = train_x(af::span, af::span, 0, af::seq(0, VAL_SIZE - 1));
    auto val_x = train_x(af::span, af::seq(0, VAL_SIZE - 1), af::span, af::span);
    std::cout << " c " << std::endl;
    train_x = train_x(af::span, af::seq(VAL_SIZE, TRAIN_SIZE - 1), af::span, af::span);
    std::cout << " c " << std::endl;
    auto val_y = train_y(af::seq(0, VAL_SIZE - 1));
    std::cout << " c " << std::endl;
    train_y = train_y(af::seq(VAL_SIZE, TRAIN_SIZE - 1));

    std::cout << " b b " << std::endl;

    fl::SGDOptimizer opt(model_->params(), learning_rate_);

    // Make the training batch dataset
    fl::BatchDataset trainset(
        std::make_shared<fl::TensorDataset>(std::vector<af::array>{train_x, train_y}),
        batch_size_);

    // Make the validation batch dataset
    fl::BatchDataset valset(
        std::make_shared<fl::TensorDataset>(std::vector<af::array>{train_x, train_y}),
        batch_size_);

    std::cout << train_x.dims() << std::endl;

    for (int e=0; e<epochs; e++){

        std::cout << "epoch " << e << std::endl;

        fl::AverageValueMeter train_loss_meter;

        //for (RowVector* it : in_dat){
        for (auto& example : trainset){
            // Make a Variable from the input array.
            auto inputs = fl::noGrad(example[INPUT_IDX]);

            // Get the activations from the model.
            fl::Sequential model = *model_;
            auto output = model(inputs);

            std::cout << "output computed " << std::endl;
            af_print(output.array());

            // Make a Variable from the target array.
            auto target = fl::noGrad(example[TARGET_IDX]);
            
            std::cout << "target " << std::endl;
            af_print(target.array());

            // Compute and record the loss.
            auto loss = fl::categoricalCrossEntropy(output, target);
            std::cout << "loss " << std::endl;
            train_loss_meter.add(loss.array().scalar<float>());
            std::cout << "train_loss_meter" << std::endl;

            // Backprop, update the weights and then zero the gradients.
            loss.backward();
            opt.step();
            opt.zeroGrad();
        }

        /*
        for (int i=0; i<in_dat.size(); i++){
            //std::cout << it->size() << " " << it->rows() << " " << it->cols() << " " << *it << std::endl;

            fl::Sequential model = *model_;

            af::array arr1 = af_from_eigen(in_dat[i]);
            auto inputs = fl::noGrad(arr1);

            auto output = model(inputs);

            af::array arr2 = af_from_eigen(out_dat[i]);
            auto target = fl::noGrad(arr2);

            auto loss = fl::categoricalCrossEntropy(output, target);
            train_loss_meter.add(loss.array().scalar<float>());

            loss.backward();
            opt.step();
            opt.zeroGrad();
        }
        */

        double train_loss = train_loss_meter.value()[0];

        std::cout << train_loss << std::endl;

        // Evaluate on the dev set.
        double val_loss, val_error;
        std::tie(val_loss, val_error) = eval_loop(*model_, valset);

        std::cout << "        Epoch " << e << std::setprecision(3)
                << ": Avg Train Loss: " << train_loss
                << " Validation Loss: " << val_loss
                << " Validation Error (%): " << val_error << std::endl;
    }

}


void agent::populate(std::string filename){
    //genData(filename);
    //ReadCSV(filename + "-in", in_dat);
    //ReadCSV(filename + "-out", out_dat);
    //std::cout << "in-out sizes = " << in_dat.size() << " " << out_dat.size() << std::endl;

    std::cout << " a " << std::endl;
    float hp[] = {2,1};
    std::cout << " a " << std::endl;
    const int nSamples = 10000;
    std::cout << " a " << std::endl;
    const int nFeat = 2;
    std::cout << " a " << std::endl;

    train_x = af::randu(nFeat, nSamples) + 1; // X elements in [1, 2]
    train_y = af::constant(0, nSamples);

    //std::cout << train_x.dims() << " " << train_y.dims() << std::endl;
    for (int i=0; i<nSamples; i++){
        //std::cout << i << std::endl;
        train_y(i) = 2 * train_x(0,i) + train_x(1,i) + 10;
    }

    std::cout << " b " << std::endl;
}




af::array agent::af_from_eigen(Eigen::RowVectorXf* v){
    Eigen::RowVectorXf v1 = *v;
    af::array arr = af::constant(0, v1.size());
    for (int i=0; i<v1.size(); i++){
        arr(i) = v1(i);
    }
    //af_print(arr);
    return arr;
}


std::pair<double, double> agent::eval_loop(fl::Sequential& model, fl::BatchDataset& dataset) {
    fl::AverageValueMeter loss_meter;
    fl::FrameErrorMeter error_meter;

    // Place the model in eval mode.
    model.eval();
    for (auto& example : dataset) {
        auto inputs = fl::noGrad(example[INPUT_IDX]);
        auto output = model(inputs);

        // Get the predictions in max_ids
        af::array max_vals, max_ids;
        max(max_vals, max_ids, output.array(), 0);

        auto target = fl::noGrad(example[TARGET_IDX]);

        // Compute and record the prediction error.
        error_meter.add(reorder(max_ids, 1, 0), target.array());

        // Compute and record the loss.
        auto loss = categoricalCrossEntropy(output, target);
        loss_meter.add(loss.array().scalar<float>());
    }
    // Place the model back into train mode.
    model.train();

    double error = error_meter.value();
    double loss = loss_meter.value()[0];
    return std::make_pair(loss, error);
}

