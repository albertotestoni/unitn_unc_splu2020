import csv

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-predictions_filename", type=str, default="../lxmert_scratch_small_predictions.csv")
    parser.add_argument("-annotations_filename", type=str, default="../locationq_classification.csv")
    parser.add_argument("-output_filename", type=str, default="../lxmert_scratch_small_predictions_annotations.csv")
    args = parser.parse_args()

    annotations = {}
    with open(args.annotations_filename) as in_file:
        reader = csv.reader(in_file)
        for row in reader:
            annotations[(row[0], row[2])] = row

    with open(args.predictions_filename) as predictions_file:
        with open(args.output_filename, mode="w") as output_file:
            predictions_reader = csv.reader(predictions_file)
            output_writer = csv.writer(output_file)
            output_writer.writerow(
                [
                    "Game ID",
                    "Position",
                    "Image",
                    "Question",
                    "GT Answer",
                    "Model Answer",
                    "Location Type",
                    "Categories"
                ]
            )
            for predictions_row in predictions_reader:
                if (predictions_row[0], predictions_row[1]) in annotations:
                    predictions_annotations = annotations[(predictions_row[0], predictions_row[1])]
                    output_writer.writerow(
                        [
                            predictions_row[0],
                            predictions_row[1],
                            predictions_row[2],
                            predictions_row[3],
                            predictions_row[4],
                            predictions_row[5],
                            predictions_annotations[3],
                            predictions_annotations[4]
                        ]
                    )
