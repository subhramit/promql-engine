window.BENCHMARK_DATA = {
  "lastUpdate": 1668071121843,
  "repoUrl": "https://github.com/thanos-community/promql-engine",
  "entries": {
    "Go Benchmark": [
      {
        "commit": {
          "author": {
            "email": "benye@amazon.com",
            "name": "Ben Ye",
            "username": "yeya24"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "075be8b8efd7d992700ae770c7126028bb82963a",
          "message": "Add continuous benchmark action for the new engine (#117)\n\n* add continuous benchmark action\r\n\r\nSigned-off-by: Ben Ye <benye@amazon.com>\r\n\r\n* remove pr trigger\r\n\r\nSigned-off-by: Ben Ye <benye@amazon.com>\r\n\r\nSigned-off-by: Ben Ye <benye@amazon.com>",
          "timestamp": "2022-11-10T08:42:00+01:00",
          "tree_id": "c2f14d11d1e2ffa2d19d3433442955ff7550301f",
          "url": "https://github.com/thanos-community/promql-engine/commit/075be8b8efd7d992700ae770c7126028bb82963a"
        },
        "date": 1668066347326,
        "tool": "go",
        "benches": [
          {
            "name": "BenchmarkRangeQuery/vector_selector",
            "value": 90607164,
            "unit": "ns/op\t28507064 B/op\t  126603 allocs/op",
            "extra": "15 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/vector_selector",
            "value": 83685365,
            "unit": "ns/op\t28690147 B/op\t  126620 allocs/op",
            "extra": "13 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/vector_selector",
            "value": 83288999,
            "unit": "ns/op\t28652094 B/op\t  126616 allocs/op",
            "extra": "13 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/vector_selector",
            "value": 83263237,
            "unit": "ns/op\t28637130 B/op\t  126612 allocs/op",
            "extra": "14 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/vector_selector",
            "value": 82696248,
            "unit": "ns/op\t28641080 B/op\t  126614 allocs/op",
            "extra": "14 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/sum",
            "value": 74052484,
            "unit": "ns/op\t 9357205 B/op\t  121230 allocs/op",
            "extra": "16 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/sum",
            "value": 72996408,
            "unit": "ns/op\t 9365085 B/op\t  121237 allocs/op",
            "extra": "15 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/sum",
            "value": 75735980,
            "unit": "ns/op\t 9367476 B/op\t  121239 allocs/op",
            "extra": "16 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/sum",
            "value": 76536613,
            "unit": "ns/op\t 9358708 B/op\t  121233 allocs/op",
            "extra": "14 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/sum",
            "value": 78393584,
            "unit": "ns/op\t 9427754 B/op\t  121241 allocs/op",
            "extra": "15 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/sum_by_pod",
            "value": 87759106,
            "unit": "ns/op\t18838061 B/op\t  206336 allocs/op",
            "extra": "13 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/sum_by_pod",
            "value": 84488013,
            "unit": "ns/op\t18524776 B/op\t  206309 allocs/op",
            "extra": "13 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/sum_by_pod",
            "value": 80909926,
            "unit": "ns/op\t18553466 B/op\t  206312 allocs/op",
            "extra": "14 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/sum_by_pod",
            "value": 78372855,
            "unit": "ns/op\t18636582 B/op\t  206314 allocs/op",
            "extra": "14 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/sum_by_pod",
            "value": 80461787,
            "unit": "ns/op\t18530427 B/op\t  206302 allocs/op",
            "extra": "14 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/rate",
            "value": 172141229,
            "unit": "ns/op\t30010712 B/op\t  150597 allocs/op",
            "extra": "7 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/rate",
            "value": 168815712,
            "unit": "ns/op\t30169733 B/op\t  150607 allocs/op",
            "extra": "6 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/rate",
            "value": 165911914,
            "unit": "ns/op\t30431227 B/op\t  150633 allocs/op",
            "extra": "7 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/rate",
            "value": 170037202,
            "unit": "ns/op\t30163411 B/op\t  150611 allocs/op",
            "extra": "7 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/rate",
            "value": 166416158,
            "unit": "ns/op\t30200006 B/op\t  150606 allocs/op",
            "extra": "7 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/sum_rate",
            "value": 168324519,
            "unit": "ns/op\t11249897 B/op\t  145256 allocs/op",
            "extra": "6 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/sum_rate",
            "value": 157299347,
            "unit": "ns/op\t11320276 B/op\t  145269 allocs/op",
            "extra": "7 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/sum_rate",
            "value": 154200563,
            "unit": "ns/op\t11239328 B/op\t  145250 allocs/op",
            "extra": "7 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/sum_rate",
            "value": 156847008,
            "unit": "ns/op\t11395478 B/op\t  145275 allocs/op",
            "extra": "7 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/sum_rate",
            "value": 162435380,
            "unit": "ns/op\t11261581 B/op\t  145264 allocs/op",
            "extra": "7 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/sum_by_rate",
            "value": 174250048,
            "unit": "ns/op\t20253628 B/op\t  230323 allocs/op",
            "extra": "6 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/sum_by_rate",
            "value": 170864042,
            "unit": "ns/op\t20212916 B/op\t  230310 allocs/op",
            "extra": "6 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/sum_by_rate",
            "value": 164516152,
            "unit": "ns/op\t20251912 B/op\t  230326 allocs/op",
            "extra": "7 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/sum_by_rate",
            "value": 176143553,
            "unit": "ns/op\t20253594 B/op\t  230323 allocs/op",
            "extra": "6 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/sum_by_rate",
            "value": 177152090,
            "unit": "ns/op\t20238636 B/op\t  230321 allocs/op",
            "extra": "6 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/binary_operation_with_one_to_one",
            "value": 44550375,
            "unit": "ns/op\t14778209 B/op\t   98308 allocs/op",
            "extra": "25 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/binary_operation_with_one_to_one",
            "value": 43066960,
            "unit": "ns/op\t14793324 B/op\t   98318 allocs/op",
            "extra": "28 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/binary_operation_with_one_to_one",
            "value": 44659512,
            "unit": "ns/op\t14768235 B/op\t   98306 allocs/op",
            "extra": "25 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/binary_operation_with_one_to_one",
            "value": 43917566,
            "unit": "ns/op\t14770283 B/op\t   98308 allocs/op",
            "extra": "26 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/binary_operation_with_one_to_one",
            "value": 44918726,
            "unit": "ns/op\t14782277 B/op\t   98312 allocs/op",
            "extra": "26 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/binary_operation_with_many_to_one",
            "value": 105935433,
            "unit": "ns/op\t35072074 B/op\t  191932 allocs/op",
            "extra": "10 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/binary_operation_with_many_to_one",
            "value": 106952454,
            "unit": "ns/op\t35036952 B/op\t  191921 allocs/op",
            "extra": "12 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/binary_operation_with_many_to_one",
            "value": 108542326,
            "unit": "ns/op\t35141700 B/op\t  191954 allocs/op",
            "extra": "10 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/binary_operation_with_many_to_one",
            "value": 108210446,
            "unit": "ns/op\t35101700 B/op\t  191969 allocs/op",
            "extra": "10 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/binary_operation_with_many_to_one",
            "value": 103997397,
            "unit": "ns/op\t35039468 B/op\t  191935 allocs/op",
            "extra": "10 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/binary_operation_with_vector_and_scalar",
            "value": 89642889,
            "unit": "ns/op\t30861498 B/op\t  130577 allocs/op",
            "extra": "13 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/binary_operation_with_vector_and_scalar",
            "value": 89362995,
            "unit": "ns/op\t30805712 B/op\t  130567 allocs/op",
            "extra": "13 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/binary_operation_with_vector_and_scalar",
            "value": 87706889,
            "unit": "ns/op\t30843471 B/op\t  130570 allocs/op",
            "extra": "13 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/binary_operation_with_vector_and_scalar",
            "value": 88439591,
            "unit": "ns/op\t30827574 B/op\t  130562 allocs/op",
            "extra": "13 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/binary_operation_with_vector_and_scalar",
            "value": 86761672,
            "unit": "ns/op\t30828638 B/op\t  130569 allocs/op",
            "extra": "13 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/unary_negation",
            "value": 88537034,
            "unit": "ns/op\t29989016 B/op\t  138950 allocs/op",
            "extra": "13 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/unary_negation",
            "value": 89864243,
            "unit": "ns/op\t29990747 B/op\t  138947 allocs/op",
            "extra": "13 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/unary_negation",
            "value": 89455699,
            "unit": "ns/op\t29990991 B/op\t  138950 allocs/op",
            "extra": "13 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/unary_negation",
            "value": 87799724,
            "unit": "ns/op\t29975545 B/op\t  138945 allocs/op",
            "extra": "13 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/unary_negation",
            "value": 88403524,
            "unit": "ns/op\t29996694 B/op\t  138950 allocs/op",
            "extra": "13 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/vector_and_scalar_comparison",
            "value": 93202495,
            "unit": "ns/op\t30488533 B/op\t  127561 allocs/op",
            "extra": "13 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/vector_and_scalar_comparison",
            "value": 93477796,
            "unit": "ns/op\t30542570 B/op\t  127576 allocs/op",
            "extra": "13 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/vector_and_scalar_comparison",
            "value": 87594949,
            "unit": "ns/op\t30482528 B/op\t  127556 allocs/op",
            "extra": "13 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/vector_and_scalar_comparison",
            "value": 86489706,
            "unit": "ns/op\t30499249 B/op\t  127557 allocs/op",
            "extra": "13 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/vector_and_scalar_comparison",
            "value": 89755646,
            "unit": "ns/op\t30479331 B/op\t  127549 allocs/op",
            "extra": "13 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/positive_offset_vector",
            "value": 75483291,
            "unit": "ns/op\t26887844 B/op\t   97820 allocs/op",
            "extra": "15 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/positive_offset_vector",
            "value": 76047537,
            "unit": "ns/op\t26844998 B/op\t   97815 allocs/op",
            "extra": "16 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/positive_offset_vector",
            "value": 78223212,
            "unit": "ns/op\t26852147 B/op\t   97817 allocs/op",
            "extra": "14 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/positive_offset_vector",
            "value": 79863062,
            "unit": "ns/op\t26843904 B/op\t   97813 allocs/op",
            "extra": "14 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/positive_offset_vector",
            "value": 77640946,
            "unit": "ns/op\t26834658 B/op\t   97809 allocs/op",
            "extra": "14 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/at_modifier_",
            "value": 55018406,
            "unit": "ns/op\t35146689 B/op\t   75419 allocs/op",
            "extra": "22 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/at_modifier_",
            "value": 50129574,
            "unit": "ns/op\t35147163 B/op\t   75420 allocs/op",
            "extra": "21 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/at_modifier_",
            "value": 51875835,
            "unit": "ns/op\t35147714 B/op\t   75421 allocs/op",
            "extra": "24 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/at_modifier_",
            "value": 51170024,
            "unit": "ns/op\t35146489 B/op\t   75418 allocs/op",
            "extra": "22 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/at_modifier_",
            "value": 50245193,
            "unit": "ns/op\t35146819 B/op\t   75419 allocs/op",
            "extra": "21 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/at_modifier_with_positive_offset_vector",
            "value": 48767472,
            "unit": "ns/op\t34957208 B/op\t   69421 allocs/op",
            "extra": "24 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/at_modifier_with_positive_offset_vector",
            "value": 48625159,
            "unit": "ns/op\t34958350 B/op\t   69420 allocs/op",
            "extra": "25 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/at_modifier_with_positive_offset_vector",
            "value": 46775422,
            "unit": "ns/op\t34957066 B/op\t   69420 allocs/op",
            "extra": "22 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/at_modifier_with_positive_offset_vector",
            "value": 50218198,
            "unit": "ns/op\t34960285 B/op\t   69421 allocs/op",
            "extra": "25 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/at_modifier_with_positive_offset_vector",
            "value": 48494150,
            "unit": "ns/op\t34958645 B/op\t   69421 allocs/op",
            "extra": "25 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/clamp",
            "value": 97101241,
            "unit": "ns/op\t29049876 B/op\t  130327 allocs/op",
            "extra": "12 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/clamp",
            "value": 97436545,
            "unit": "ns/op\t29101162 B/op\t  130329 allocs/op",
            "extra": "12 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/clamp",
            "value": 98189612,
            "unit": "ns/op\t29051492 B/op\t  130322 allocs/op",
            "extra": "12 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/clamp",
            "value": 101274241,
            "unit": "ns/op\t29051734 B/op\t  130337 allocs/op",
            "extra": "12 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/clamp",
            "value": 106144260,
            "unit": "ns/op\t29052615 B/op\t  130327 allocs/op",
            "extra": "12 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/clamp_min",
            "value": 105334573,
            "unit": "ns/op\t29091496 B/op\t  129993 allocs/op",
            "extra": "12 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/clamp_min",
            "value": 102009158,
            "unit": "ns/op\t29033092 B/op\t  129980 allocs/op",
            "extra": "12 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/clamp_min",
            "value": 101385045,
            "unit": "ns/op\t29043477 B/op\t  129984 allocs/op",
            "extra": "12 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/clamp_min",
            "value": 100770615,
            "unit": "ns/op\t29037333 B/op\t  129986 allocs/op",
            "extra": "13 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/clamp_min",
            "value": 100345127,
            "unit": "ns/op\t29104281 B/op\t  129995 allocs/op",
            "extra": "12 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/complex_func_query",
            "value": 110533858,
            "unit": "ns/op\t31127189 B/op\t  135027 allocs/op",
            "extra": "10 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/complex_func_query",
            "value": 115552516,
            "unit": "ns/op\t31156940 B/op\t  135031 allocs/op",
            "extra": "9 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/complex_func_query",
            "value": 111480132,
            "unit": "ns/op\t31129855 B/op\t  135021 allocs/op",
            "extra": "10 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/complex_func_query",
            "value": 111608992,
            "unit": "ns/op\t31131043 B/op\t  135023 allocs/op",
            "extra": "9 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/complex_func_query",
            "value": 117587124,
            "unit": "ns/op\t31191423 B/op\t  135037 allocs/op",
            "extra": "10 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/func_within_func_query",
            "value": 158082239,
            "unit": "ns/op\t30156021 B/op\t  152053 allocs/op",
            "extra": "7 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/func_within_func_query",
            "value": 160199610,
            "unit": "ns/op\t30238349 B/op\t  152061 allocs/op",
            "extra": "7 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/func_within_func_query",
            "value": 159464674,
            "unit": "ns/op\t30239870 B/op\t  152065 allocs/op",
            "extra": "7 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/func_within_func_query",
            "value": 159827206,
            "unit": "ns/op\t30247370 B/op\t  152070 allocs/op",
            "extra": "7 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/func_within_func_query",
            "value": 152680503,
            "unit": "ns/op\t30312305 B/op\t  152068 allocs/op",
            "extra": "7 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/aggr_within_func_query",
            "value": 172902985,
            "unit": "ns/op\t30217400 B/op\t  152050 allocs/op",
            "extra": "7 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/aggr_within_func_query",
            "value": 177635927,
            "unit": "ns/op\t30172422 B/op\t  152057 allocs/op",
            "extra": "6 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/aggr_within_func_query",
            "value": 175125741,
            "unit": "ns/op\t30272178 B/op\t  152077 allocs/op",
            "extra": "6 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/aggr_within_func_query",
            "value": 166003353,
            "unit": "ns/op\t30218893 B/op\t  152054 allocs/op",
            "extra": "7 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/aggr_within_func_query",
            "value": 162300206,
            "unit": "ns/op\t30302765 B/op\t  152066 allocs/op",
            "extra": "7 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/histogram_quantile",
            "value": 301917590,
            "unit": "ns/op\t96749286 B/op\t  701247 allocs/op",
            "extra": "4 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/histogram_quantile",
            "value": 306212944,
            "unit": "ns/op\t97378054 B/op\t  701240 allocs/op",
            "extra": "4 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/histogram_quantile",
            "value": 311903444,
            "unit": "ns/op\t96971906 B/op\t  701230 allocs/op",
            "extra": "4 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/histogram_quantile",
            "value": 327200156,
            "unit": "ns/op\t96340118 B/op\t  701214 allocs/op",
            "extra": "4 times\n2 procs"
          },
          {
            "name": "BenchmarkRangeQuery/histogram_quantile",
            "value": 307777858,
            "unit": "ns/op\t96954848 B/op\t  701240 allocs/op",
            "extra": "4 times\n2 procs"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "giedrius.statkevicius@vinted.com",
            "name": "Giedrius Statkevičius",
            "username": "GiedriusS"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "437e914ef890465cb689b323e3540bb8d9ba3432",
          "message": ".github: use self-hosted runner (#118)\n\nI started a small runner on Equinix using CNCF's resources\r\n(c3.small.x86). Let's use it to have consistent results.",
          "timestamp": "2022-11-10T10:58:06+02:00",
          "tree_id": "e2ad364a66ee991f439f0dd266b1f98b51364171",
          "url": "https://github.com/thanos-community/promql-engine/commit/437e914ef890465cb689b323e3540bb8d9ba3432"
        },
        "date": 1668071121175,
        "tool": "go",
        "benches": [
          {
            "name": "BenchmarkRangeQuery/vector_selector",
            "value": 23120225,
            "unit": "ns/op\t29488801 B/op\t  131575 allocs/op",
            "extra": "50 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/vector_selector",
            "value": 23457147,
            "unit": "ns/op\t29484337 B/op\t  131553 allocs/op",
            "extra": "48 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/vector_selector",
            "value": 23411400,
            "unit": "ns/op\t29498829 B/op\t  131572 allocs/op",
            "extra": "49 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/vector_selector",
            "value": 23510383,
            "unit": "ns/op\t29497642 B/op\t  131567 allocs/op",
            "extra": "49 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/vector_selector",
            "value": 24345525,
            "unit": "ns/op\t29504788 B/op\t  131566 allocs/op",
            "extra": "49 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/sum",
            "value": 11015925,
            "unit": "ns/op\t12254050 B/op\t  126254 allocs/op",
            "extra": "100 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/sum",
            "value": 11051543,
            "unit": "ns/op\t12215861 B/op\t  126231 allocs/op",
            "extra": "100 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/sum",
            "value": 11032129,
            "unit": "ns/op\t12204306 B/op\t  126221 allocs/op",
            "extra": "100 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/sum",
            "value": 11049268,
            "unit": "ns/op\t12225254 B/op\t  126221 allocs/op",
            "extra": "100 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/sum",
            "value": 10995732,
            "unit": "ns/op\t12201526 B/op\t  126233 allocs/op",
            "extra": "100 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/sum_by_pod",
            "value": 20434183,
            "unit": "ns/op\t21043885 B/op\t  211879 allocs/op",
            "extra": "56 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/sum_by_pod",
            "value": 20599917,
            "unit": "ns/op\t21064398 B/op\t  211888 allocs/op",
            "extra": "57 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/sum_by_pod",
            "value": 20481112,
            "unit": "ns/op\t21031938 B/op\t  211874 allocs/op",
            "extra": "56 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/sum_by_pod",
            "value": 20619206,
            "unit": "ns/op\t21043267 B/op\t  211880 allocs/op",
            "extra": "57 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/sum_by_pod",
            "value": 20639628,
            "unit": "ns/op\t21019515 B/op\t  211859 allocs/op",
            "extra": "57 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/rate",
            "value": 29593339,
            "unit": "ns/op\t31304588 B/op\t  155333 allocs/op",
            "extra": "39 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/rate",
            "value": 29659619,
            "unit": "ns/op\t31278144 B/op\t  155336 allocs/op",
            "extra": "38 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/rate",
            "value": 29695246,
            "unit": "ns/op\t31320430 B/op\t  155379 allocs/op",
            "extra": "38 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/rate",
            "value": 29528457,
            "unit": "ns/op\t31287905 B/op\t  155344 allocs/op",
            "extra": "40 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/rate",
            "value": 29584446,
            "unit": "ns/op\t31291036 B/op\t  155327 allocs/op",
            "extra": "40 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/sum_rate",
            "value": 21657878,
            "unit": "ns/op\t13694550 B/op\t  149968 allocs/op",
            "extra": "52 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/sum_rate",
            "value": 21711615,
            "unit": "ns/op\t13775519 B/op\t  150055 allocs/op",
            "extra": "54 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/sum_rate",
            "value": 21725486,
            "unit": "ns/op\t13668291 B/op\t  149962 allocs/op",
            "extra": "52 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/sum_rate",
            "value": 21566924,
            "unit": "ns/op\t13687928 B/op\t  149969 allocs/op",
            "extra": "54 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/sum_rate",
            "value": 21683936,
            "unit": "ns/op\t13693237 B/op\t  150020 allocs/op",
            "extra": "52 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/sum_by_rate",
            "value": 30040968,
            "unit": "ns/op\t21907778 B/op\t  235327 allocs/op",
            "extra": "39 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/sum_by_rate",
            "value": 29814989,
            "unit": "ns/op\t21937274 B/op\t  235357 allocs/op",
            "extra": "38 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/sum_by_rate",
            "value": 29772861,
            "unit": "ns/op\t21857331 B/op\t  235292 allocs/op",
            "extra": "38 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/sum_by_rate",
            "value": 30086868,
            "unit": "ns/op\t21838980 B/op\t  235278 allocs/op",
            "extra": "39 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/sum_by_rate",
            "value": 29986858,
            "unit": "ns/op\t21821218 B/op\t  235247 allocs/op",
            "extra": "38 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/binary_operation_with_one_to_one",
            "value": 18798030,
            "unit": "ns/op\t16013780 B/op\t  107820 allocs/op",
            "extra": "61 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/binary_operation_with_one_to_one",
            "value": 18833749,
            "unit": "ns/op\t16015921 B/op\t  107821 allocs/op",
            "extra": "62 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/binary_operation_with_one_to_one",
            "value": 18808770,
            "unit": "ns/op\t16022963 B/op\t  107840 allocs/op",
            "extra": "62 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/binary_operation_with_one_to_one",
            "value": 18718612,
            "unit": "ns/op\t16024458 B/op\t  107858 allocs/op",
            "extra": "60 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/binary_operation_with_one_to_one",
            "value": 18783393,
            "unit": "ns/op\t16039210 B/op\t  107885 allocs/op",
            "extra": "64 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/binary_operation_with_many_to_one",
            "value": 39876289,
            "unit": "ns/op\t36991008 B/op\t  201925 allocs/op",
            "extra": "30 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/binary_operation_with_many_to_one",
            "value": 39869145,
            "unit": "ns/op\t37024094 B/op\t  201951 allocs/op",
            "extra": "30 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/binary_operation_with_many_to_one",
            "value": 40194021,
            "unit": "ns/op\t37058511 B/op\t  201972 allocs/op",
            "extra": "28 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/binary_operation_with_many_to_one",
            "value": 39968489,
            "unit": "ns/op\t36973794 B/op\t  201908 allocs/op",
            "extra": "28 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/binary_operation_with_many_to_one",
            "value": 39875511,
            "unit": "ns/op\t37001806 B/op\t  201910 allocs/op",
            "extra": "30 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/binary_operation_with_vector_and_scalar",
            "value": 33661697,
            "unit": "ns/op\t33110541 B/op\t  135933 allocs/op",
            "extra": "36 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/binary_operation_with_vector_and_scalar",
            "value": 33652740,
            "unit": "ns/op\t33082099 B/op\t  135901 allocs/op",
            "extra": "36 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/binary_operation_with_vector_and_scalar",
            "value": 33333358,
            "unit": "ns/op\t33113063 B/op\t  135925 allocs/op",
            "extra": "34 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/binary_operation_with_vector_and_scalar",
            "value": 33933509,
            "unit": "ns/op\t33069324 B/op\t  135894 allocs/op",
            "extra": "36 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/binary_operation_with_vector_and_scalar",
            "value": 33533124,
            "unit": "ns/op\t33066108 B/op\t  135900 allocs/op",
            "extra": "36 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/unary_negation",
            "value": 25863318,
            "unit": "ns/op\t30777038 B/op\t  143839 allocs/op",
            "extra": "45 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/unary_negation",
            "value": 25832540,
            "unit": "ns/op\t30791259 B/op\t  143853 allocs/op",
            "extra": "48 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/unary_negation",
            "value": 25691176,
            "unit": "ns/op\t30801793 B/op\t  143869 allocs/op",
            "extra": "48 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/unary_negation",
            "value": 25718918,
            "unit": "ns/op\t30774227 B/op\t  143848 allocs/op",
            "extra": "45 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/unary_negation",
            "value": 25749177,
            "unit": "ns/op\t30788227 B/op\t  143860 allocs/op",
            "extra": "46 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/vector_and_scalar_comparison",
            "value": 33637429,
            "unit": "ns/op\t32762307 B/op\t  132908 allocs/op",
            "extra": "34 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/vector_and_scalar_comparison",
            "value": 34047802,
            "unit": "ns/op\t32760843 B/op\t  132907 allocs/op",
            "extra": "36 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/vector_and_scalar_comparison",
            "value": 33977428,
            "unit": "ns/op\t32673860 B/op\t  132864 allocs/op",
            "extra": "34 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/vector_and_scalar_comparison",
            "value": 33776724,
            "unit": "ns/op\t32867111 B/op\t  132974 allocs/op",
            "extra": "34 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/vector_and_scalar_comparison",
            "value": 33858184,
            "unit": "ns/op\t32786131 B/op\t  132914 allocs/op",
            "extra": "34 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/positive_offset_vector",
            "value": 23901868,
            "unit": "ns/op\t27742519 B/op\t  102603 allocs/op",
            "extra": "51 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/positive_offset_vector",
            "value": 23766700,
            "unit": "ns/op\t27755439 B/op\t  102617 allocs/op",
            "extra": "51 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/positive_offset_vector",
            "value": 23642250,
            "unit": "ns/op\t27745230 B/op\t  102621 allocs/op",
            "extra": "52 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/positive_offset_vector",
            "value": 23954648,
            "unit": "ns/op\t27755001 B/op\t  102630 allocs/op",
            "extra": "48 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/positive_offset_vector",
            "value": 23734406,
            "unit": "ns/op\t27753439 B/op\t  102614 allocs/op",
            "extra": "49 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/at_modifier_",
            "value": 21002522,
            "unit": "ns/op\t35236576 B/op\t   75784 allocs/op",
            "extra": "58 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/at_modifier_",
            "value": 21053377,
            "unit": "ns/op\t35237572 B/op\t   75786 allocs/op",
            "extra": "54 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/at_modifier_",
            "value": 21609766,
            "unit": "ns/op\t35237264 B/op\t   75786 allocs/op",
            "extra": "62 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/at_modifier_",
            "value": 21545322,
            "unit": "ns/op\t35236661 B/op\t   75786 allocs/op",
            "extra": "52 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/at_modifier_",
            "value": 21359420,
            "unit": "ns/op\t35236762 B/op\t   75785 allocs/op",
            "extra": "61 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/at_modifier_with_positive_offset_vector",
            "value": 20755403,
            "unit": "ns/op\t35059434 B/op\t   69786 allocs/op",
            "extra": "64 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/at_modifier_with_positive_offset_vector",
            "value": 20983246,
            "unit": "ns/op\t35061921 B/op\t   69787 allocs/op",
            "extra": "55 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/at_modifier_with_positive_offset_vector",
            "value": 20897973,
            "unit": "ns/op\t35057679 B/op\t   69787 allocs/op",
            "extra": "62 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/at_modifier_with_positive_offset_vector",
            "value": 20525502,
            "unit": "ns/op\t35054588 B/op\t   69785 allocs/op",
            "extra": "64 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/at_modifier_with_positive_offset_vector",
            "value": 21055453,
            "unit": "ns/op\t35059278 B/op\t   69786 allocs/op",
            "extra": "61 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/clamp",
            "value": 41746183,
            "unit": "ns/op\t29871783 B/op\t  135385 allocs/op",
            "extra": "27 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/clamp",
            "value": 41912288,
            "unit": "ns/op\t29847896 B/op\t  135352 allocs/op",
            "extra": "27 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/clamp",
            "value": 42167788,
            "unit": "ns/op\t29864284 B/op\t  135402 allocs/op",
            "extra": "27 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/clamp",
            "value": 41680461,
            "unit": "ns/op\t29871951 B/op\t  135400 allocs/op",
            "extra": "27 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/clamp",
            "value": 42312623,
            "unit": "ns/op\t29876883 B/op\t  135369 allocs/op",
            "extra": "28 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/clamp_min",
            "value": 39354988,
            "unit": "ns/op\t29806071 B/op\t  134933 allocs/op",
            "extra": "30 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/clamp_min",
            "value": 38620371,
            "unit": "ns/op\t29794921 B/op\t  134928 allocs/op",
            "extra": "30 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/clamp_min",
            "value": 38583010,
            "unit": "ns/op\t29826015 B/op\t  134944 allocs/op",
            "extra": "31 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/clamp_min",
            "value": 38515376,
            "unit": "ns/op\t29825331 B/op\t  134948 allocs/op",
            "extra": "31 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/clamp_min",
            "value": 38819711,
            "unit": "ns/op\t29814630 B/op\t  134927 allocs/op",
            "extra": "31 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/complex_func_query",
            "value": 50060373,
            "unit": "ns/op\t33351743 B/op\t  140395 allocs/op",
            "extra": "22 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/complex_func_query",
            "value": 50539257,
            "unit": "ns/op\t33279884 B/op\t  140344 allocs/op",
            "extra": "22 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/complex_func_query",
            "value": 50649788,
            "unit": "ns/op\t33344264 B/op\t  140378 allocs/op",
            "extra": "22 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/complex_func_query",
            "value": 50401350,
            "unit": "ns/op\t33384753 B/op\t  140417 allocs/op",
            "extra": "22 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/complex_func_query",
            "value": 50654603,
            "unit": "ns/op\t33304611 B/op\t  140374 allocs/op",
            "extra": "22 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/func_within_func_query",
            "value": 42314541,
            "unit": "ns/op\t31482978 B/op\t  157024 allocs/op",
            "extra": "28 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/func_within_func_query",
            "value": 42349805,
            "unit": "ns/op\t31488789 B/op\t  157046 allocs/op",
            "extra": "27 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/func_within_func_query",
            "value": 42417056,
            "unit": "ns/op\t31439482 B/op\t  156998 allocs/op",
            "extra": "27 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/func_within_func_query",
            "value": 42070619,
            "unit": "ns/op\t31488926 B/op\t  157049 allocs/op",
            "extra": "27 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/func_within_func_query",
            "value": 42431864,
            "unit": "ns/op\t31453563 B/op\t  157007 allocs/op",
            "extra": "27 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/aggr_within_func_query",
            "value": 42421780,
            "unit": "ns/op\t31488109 B/op\t  157042 allocs/op",
            "extra": "27 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/aggr_within_func_query",
            "value": 42244641,
            "unit": "ns/op\t31508149 B/op\t  157054 allocs/op",
            "extra": "28 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/aggr_within_func_query",
            "value": 42444444,
            "unit": "ns/op\t31481559 B/op\t  157048 allocs/op",
            "extra": "27 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/aggr_within_func_query",
            "value": 42433890,
            "unit": "ns/op\t31452712 B/op\t  156993 allocs/op",
            "extra": "27 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/aggr_within_func_query",
            "value": 42408842,
            "unit": "ns/op\t31441933 B/op\t  156995 allocs/op",
            "extra": "27 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/histogram_quantile",
            "value": 130255654,
            "unit": "ns/op\t98568157 B/op\t  704813 allocs/op",
            "extra": "8 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/histogram_quantile",
            "value": 132319127,
            "unit": "ns/op\t98456460 B/op\t  704742 allocs/op",
            "extra": "9 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/histogram_quantile",
            "value": 131858084,
            "unit": "ns/op\t98567417 B/op\t  704814 allocs/op",
            "extra": "8 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/histogram_quantile",
            "value": 130696972,
            "unit": "ns/op\t98681143 B/op\t  704925 allocs/op",
            "extra": "9 times\n16 procs"
          },
          {
            "name": "BenchmarkRangeQuery/histogram_quantile",
            "value": 131971556,
            "unit": "ns/op\t98495217 B/op\t  704751 allocs/op",
            "extra": "8 times\n16 procs"
          }
        ]
      }
    ]
  }
}