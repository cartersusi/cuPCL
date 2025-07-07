#include "point_cloud.h"

int main() {
    cout << "Testing point cloud with max_points = " << prediss_point_cloud::max_points << " points\n";
    
    // Create point cloud
    prediss_point_cloud::point_cloud pc;
    pc.timestamp_ns = chrono::duration_cast<chrono::nanoseconds>(chrono::system_clock::now().time_since_epoch()).count();
    
    // Setup random number generator for realistic point data
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<uint32_t> coord_dist(0, 1000000); // 0-1000m in mm
    
    // Start timing
    auto start_time = chrono::high_resolution_clock::now();
    
    try {
        // Add maximum number of points
        for (int i = 0; i < prediss_point_cloud::max_points - 1; ++i) {
            uint32_t x = coord_dist(gen);
            uint32_t y = coord_dist(gen);
            uint32_t z = coord_dist(gen);
            
            pc.set_uint_point(i, prediss_point_cloud::create_raw_uint_point(x, y, z));
        }
        
        auto end_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
        
        cout << "\nSuccessfully added " << prediss_point_cloud::max_points << " points\n";
        cout << "Time taken: " << duration.count() << " ms\n";
        cout << "Points per second: " << (prediss_point_cloud::max_points * 1000.0 / duration.count()) << "\n";
        cout << "Final point cloud size: " << pc.size() << " float values\n";
        cout << "Stack Space: " << (pc.n_points * prediss_point_cloud::point_size * sizeof(float)) / (1024) << " KB\n";
        
        // Verify a few random points
        cout << "\nVerifying random points:\n";
        uniform_int_distribution<int> index_dist(0, prediss_point_cloud::max_points - 1);
        for (int i = 0; i < 5; ++i) {
            int idx = index_dist(gen);
            prediss_point_cloud::point p = pc.get_point(idx);
            cout << "Point " << idx << ": (" << p[0] << ", " << p[1] << ", " << p[2] << ")\n";
        }
        
    } catch (const exception& e) {
        auto end_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
        
        cout << "\nException occurred: " << e.what() << "\n";
        cout << "Points added before exception: " << pc.n_points << "\n";
        cout << "Time taken: " << duration.count() << " ms\n";
        return 1;
    }
    
    return 0;
}