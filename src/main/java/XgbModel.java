import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;
import tech.tablesaw.api.FloatColumn;
import tech.tablesaw.api.StringColumn;
import tech.tablesaw.api.Table;
import ml.dmlc.xgboost4j.java.Booster;

import java.util.HashMap;
import java.util.Objects;

public class XgbModel {
    public static void main(String[] args) {
        Table data = Table.read().csv("/home/hari-pt6946/workFiles/datasets/drug200.csv");
        System.out.println(data.first(5));

        StringColumn drug = data.stringColumn("Drug");

        // Label(Drug) from string value to int
        int drugIndex = 0;
        for(String drugName: data.stringColumn("Drug").unique()){
            drug.set(drug.isEqualTo(drugName), String.valueOf(drugIndex));
            drugIndex++;
        }

        // Handling Cholesterol column
        StringColumn cholesterol = data.stringColumn("Cholesterol");

        // Cholesterol from categorical to numeric(1,2,....)  HIGH=2, NORMAL=1
        int level_numeric = 2;
        for(String level: data.stringColumn("Cholesterol").unique()){
            cholesterol.set(cholesterol.isEqualTo(level), String.valueOf(level_numeric));
            level_numeric--;
        }

        FloatColumn cholesterol_int = data.stringColumn("Cholesterol").parseFloat();
        cholesterol_int.setName("Cholesterol");
        data.removeColumns("Cholesterol");
        data.addColumns(cholesterol_int);


        // Handling BP column
        var bp = data.stringColumn("BP");
        int bpLevel_numeric;
        for(String level: data.stringColumn("BP").unique()){
            if(Objects.equals(level, "HIGH")){
                bpLevel_numeric = 3;
            } else if (Objects.equals(level, "NORMAL")) {
                bpLevel_numeric = 2;
            } else {
                bpLevel_numeric = 1;
            }
            bp.set(bp.isEqualTo(level), String.valueOf(bpLevel_numeric));
        }
        FloatColumn bp_int = bp.parseFloat();
        bp_int.setName("BP");
        data.removeColumns("BP");
        data.addColumns(bp_int);


        // Handling Sex column
        StringColumn sex = data.selectColumns("Sex").stringColumn("Sex");
//        StringColumn sexM = data.selectColumns("Sex").stringColumn("Sex");
        // one hot encoding
        String numeric;
        for(String gender: sex.unique()){
            if(Objects.equals(gender, "F")){
                numeric = "1";
            } else {
                numeric = "0";
            }
            sex.set(sex.isEqualTo(gender), numeric);
        }
        FloatColumn sex_f = sex.parseFloat();
        sex_f.setName("Sex_F");
        data.removeColumns("Sex");
        data.addColumns(sex_f);

//        for(String gender: sexM.unique()){
//            if(Objects.equals(gender, "M")){
//                numeric = "1";
//            } else {
//                numeric = "0";
//            }
//            sexM.set(sexM.isEqualTo(gender), numeric);
//        }
//        FloatColumn sex_m = sexM.parseFloat();
//        sex_m.setName("Sex_M");
//        data.addColumns(sex_m);

        System.out.println(data.structure());


        // AGE to Float
        FloatColumn age_float = data.intColumn("Age").asFloatColumn();
        data.removeColumns("Age");
        age_float.setName("Age");
        data.addColumns(age_float);

        // Na_to_K from int to Float
        FloatColumn nak_float = data.doubleColumn("Na_to_K").asFloatColumn();
        data.removeColumns("Na_to_K");
        nak_float.setName("Na_to_K");
        data.addColumns(nak_float);


        // Table for Label y
        Table y = data.selectColumns("Drug");
        FloatColumn drug_int = y.stringColumn("Drug").parseFloat();
        y.removeColumns("Drug");
        drug_int.setName("labels");
        y.addColumns(drug_int);
        y.setName("Labels");
        System.out.println(y.structure());

        /* data review */
//        System.out.println(data.column("Drug").summary());
//        System.out.println(y.summary());


        // Table for feature X
        Table X = data.rejectColumns("Drug");
        X.setName("Features");
        drug_int.setName("labels");
        System.out.println(X.structure());

        int rowCount = X.rowCount();
        int trainCount = (int) (rowCount * 0.8);

        Table x_train = X.inRange(0,trainCount+1);
        x_train.setName("X Train Table");
        Table x_test = X.inRange(trainCount+1, rowCount);
        x_test.setName("X Test Table");
        Table y_train = y.inRange(0,trainCount+1);
        Table y_test = y.inRange(trainCount+1, rowCount);

        System.out.println("x_train");
        System.out.println(x_train.structure());



        DMatrix x_train_dMat, y_train_dMat, x_test_dMat;
        int x_train_rowCount = x_train.rowCount();
        int x_train_colmCount = x_train.columnCount();
        int x_test_rowCount = x_test.rowCount();
        int x_test_colCount = x_test.columnCount();


        float[][] x_train_temp = new float[x_train_rowCount][x_train_colmCount];
        float[][] x_test_temp = new float[x_test_rowCount][x_test_colCount];

        for(int rowc = 0; rowc < x_train_rowCount; rowc++){
            for(int colc = 0; colc < x_train_colmCount; colc++){
                x_train_temp[rowc][colc] = x_train.row(rowc).getFloat(colc);
            }
        }

        for(int rowc = 0; rowc < x_test_rowCount; rowc++){
            for(int colc = 0; colc < x_test_colCount; colc++){
                x_test_temp[rowc][colc] = x_test.row(rowc).getFloat(colc);
            }
        }

        float[] x_train_fArr = flatten(x_train_temp);
        float[] x_test_fArr = flatten(x_test_temp);



        float missing = Float.NaN;
        try {
            x_train_dMat = new DMatrix(x_train_fArr, x_train_rowCount, x_train_colmCount, missing);
            x_test_dMat = new DMatrix(x_test_fArr, x_test_rowCount, x_test_colCount, missing);

        } catch (XGBoostError e) {
            System.out.println("DMatrix Error");
            throw new RuntimeException(e);
        }

        try {
            x_train_dMat.setLabel(y_train.floatColumn(0).asFloatArray());
        } catch (XGBoostError e) {
            throw new RuntimeException(e);
        }


        HashMap<String, Object> params = new HashMap<>();
        params.put("eta", 0.3);
        params.put("max_depth", 5);
        params.put("num_class", 5);
        params.put("objective", "multi:softmax");


        HashMap<String, DMatrix> watches = new HashMap<>();
        watches.put("train", x_train_dMat);
        watches.put("test", x_test_dMat);


        Booster booster;
        try {
            booster = XGBoost.train(x_train_dMat, params, 100, watches, null, null);
        } catch (XGBoostError e) {
            System.out.println("ERROR LOADING MODEL");
            throw new RuntimeException(e);
        }

        float[][] y_pred;
        try {
            y_pred = booster.predict(x_test_dMat);
        } catch (XGBoostError e) {
            throw new RuntimeException(e);
        }

        for(float[] i: y_pred){
            for(float j: i){
                System.out.printf("%f ",j);
            }
            System.out.println();
        }
    }

    private static float[] flatten(float[][] mat) {
        int size = 0;
        for (float[] array : mat) size += array.length;
        float[] result = new float[size];
        int pos = 0;
        for (float[] ar : mat) {
            System.arraycopy(ar, 0, result, pos, ar.length);
            pos += ar.length;
        }

        return result;
    }

}
