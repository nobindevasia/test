using System;
using System.Collections.Generic;
using Microsoft.Data.SqlClient;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using D2G.Iris.ML.Core.Enums;
using D2G.Iris.ML.Core.Interfaces;

namespace D2G.Iris.ML.Data
{
    public class DataLoader : IDataLoader
    {
        public IEnumerable<Dictionary<string, object>> LoadDataFromSql(
            string sqlConnectionString,
            string tableName,
            IEnumerable<string> fields,
            ModelType modelType,
            string targetField,
            string whereSyntax = "")
        {
            Console.WriteLine("=============== Load Data ==============");
            var finalFields = fields.Where(f => f != targetField).ToList();
            var results = new List<Dictionary<string, object>>();

            using (var connection = new SqlConnection(sqlConnectionString))
            {
                connection.Open();

                var fieldList = finalFields.Any() ? string.Join(", ", finalFields) : "*";
                var sql = $"SELECT {fieldList}, [{targetField}] FROM {tableName}";
                if (!string.IsNullOrEmpty(whereSyntax))
                {
                    sql += $" WHERE {whereSyntax}";
                }

                using (var command = new SqlCommand(sql, connection))
                {
                    command.CommandTimeout = 120;
                    using (var reader = command.ExecuteReader())
                    {
                        while (reader.Read())
                        {
                            var record = new Dictionary<string, object>();

                            foreach (var field in fields)
                            {
                                var value = reader[field];
                                record[field] = value == DBNull.Value ? 0.0f : Convert.ToSingle(value);
                            }

                            var targetValue = reader[targetField];
                            record[targetField] = modelType switch
                            {
                                ModelType.BinaryClassification => Convert.ToInt64(targetValue) != 0,
                                ModelType.MultiClassClassification => Convert.ToUInt32(targetValue),
                                ModelType.Regression => targetValue == DBNull.Value ? 0.0f : Convert.ToSingle(targetValue),
                                _ => Convert.ToInt64(targetValue) != 0
                            };

                            results.Add(record);
                        }
                    }
                }
            }

            ValidateLoadedData(results, modelType);
            Console.WriteLine($"Loaded {results.Count} rows of Data.");
            return results;
        }

        public IDataView ConvertToDataView(
            MLContext mlContext,
            IEnumerable<Dictionary<string, object>> data,
            string[] features,
            ModelType modelType,
            string targetField)
        {
            switch (modelType)
            {
                case ModelType.BinaryClassification:
                    var binaryRows = data.Select(dict => new BinaryRow
                    {
                        Features = features.Select(field =>
                            dict[field] == null ? 0.0f : Convert.ToSingle(dict[field])).ToArray(),
                        Label = Convert.ToBoolean(dict[targetField])
                    }).ToList();

                    var binarySchema = SchemaDefinition.Create(typeof(BinaryRow));
                    binarySchema["Features"].ColumnType = new VectorDataViewType(NumberDataViewType.Single, features.Length);
                    return mlContext.Data.LoadFromEnumerable(binaryRows, binarySchema);

                case ModelType.MultiClassClassification:
                    var multiRows = data.Select(dict => new MulticlassRow
                    {
                        Features = features.Select(field =>
                            dict[field] == null ? 0.0f : Convert.ToSingle(dict[field])).ToArray(),
                        Label = Convert.ToUInt32(dict[targetField])
                    }).ToList();

                    var multiSchema = SchemaDefinition.Create(typeof(MulticlassRow));
                    multiSchema["Features"].ColumnType = new VectorDataViewType(NumberDataViewType.Single, features.Length);
                    return mlContext.Data.LoadFromEnumerable(multiRows, multiSchema);

                case ModelType.Regression:
                    var regRows = data.Select(dict => new RegressionRow
                    {
                        Features = features.Select(field =>
                            dict[field] == null ? 0.0f : Convert.ToSingle(dict[field])).ToArray(),
                        Label = Convert.ToSingle(dict[targetField])
                    }).ToList();

                    var regSchema = SchemaDefinition.Create(typeof(RegressionRow));
                    regSchema["Features"].ColumnType = new VectorDataViewType(NumberDataViewType.Single, features.Length);
                    return mlContext.Data.LoadFromEnumerable(regRows, regSchema);

                default:
                    throw new ArgumentException($"Unsupported model type: {modelType}");
            }
        }

        private void ValidateLoadedData(List<Dictionary<string, object>> data, ModelType modelType)
        {
            if (!data.Any())
                throw new InvalidOperationException("No data was loaded from the database.");

            switch (modelType)
            {
                case ModelType.BinaryClassification:
                    var binaryValues = data.Select(d => Convert.ToBoolean(d["Label"])).Distinct().ToList();
                    if (binaryValues.Count != 2)
                    {
                        throw new InvalidOperationException(
                            $"Binary classification requires exactly two distinct label values. Found {binaryValues.Count} values.");
                    }
                    break;

                case ModelType.MultiClassClassification:
                    var classLabels = data.Select(d => Convert.ToUInt32(d["Label"])).Distinct().OrderBy(x => x).ToList();
                    if (!classLabels.SequenceEqual(Enumerable.Range(0, classLabels.Count).Select(x => (uint)x)))
                    {
                        throw new InvalidOperationException(
                            "Multiclass labels must be consecutive integers starting from 0. " +
                            $"Found values: {string.Join(", ", classLabels)}");
                    }
                    break;

                case ModelType.Regression:
                    var regressionValues = data.Select(d => Convert.ToSingle(d["Label"]));
                    if (regressionValues.Any(float.IsInfinity) || regressionValues.Any(float.IsNaN))
                    {
                        throw new InvalidOperationException("Regression labels contain invalid values");
                    }                
                    break;
            }
        }

        private class BinaryRow
        {
            public float[] Features { get; set; }
            public bool Label { get; set; }
        }

        private class MulticlassRow
        {
            public float[] Features { get; set; }
            public uint Label { get; set; }
        }

        private class RegressionRow
        {
            public float[] Features { get; set; }
            public float Label { get; set; }
        }
    }
}