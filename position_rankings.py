"""Position Rankings analysis functions for Streamlit app."""

from typing import Dict, List

import pandas as pd
import streamlit as st

def render_position_rankings_section(filtered_df: pd.DataFrame, role_names: List[str], cfg: Dict) -> None:
    """Render the position rankings analysis section."""
    unique_key = cfg["database"]["unique_key"]
    st.header("Position Rankings Analysis")
    
    # Count total applications with preferences
    total_apps = len(filtered_df)
    with_prefs = len(filtered_df[filtered_df["Pref count"] > 0])
    
    col1, col2 = st.columns(2)
    col1.metric("Total Applications", total_apps)
    col2.metric("With Position Preferences", with_prefs, f"{with_prefs/max(1, total_apps):.1%}")
    
    st.subheader("Filter by Position Preferences")
    
    # Create multi-select filters for each position
    filter_cols = st.columns(len(role_names))
    
    role_filters = {}
    for i, role in enumerate(role_names):
        role_short = role.split()[0]  # Just use first word for brevity
        col_name = f"{role_short} Rank"
        
        if col_name in filtered_df.columns:
            options = ["Any", "1", "2", "3", "None"]
            role_filters[role] = filter_cols[i].selectbox(
                f"{role} Rank", 
                options=options,
                index=0,
                key=f"rank_filter_{role}"
            )
    
    # Apply filters
    filtered_by_role = filtered_df.copy()
    for role, rank_filter in role_filters.items():
        role_short = role.split()[0]
        col_name = f"{role_short} Rank"
        
        if rank_filter != "Any" and col_name in filtered_by_role.columns:
            if rank_filter == "None":
                filtered_by_role = filtered_by_role[filtered_by_role[col_name].isin(["", "nan", "none", "null"])]
            else:
                filtered_by_role = filtered_by_role[filtered_by_role[col_name] == rank_filter]
    
    # Show results
    if not filtered_by_role.empty:
        # Create a column to show position preferences
        display_cols = [
            unique_key,
            "Username",
            "First Name", 
            "Last Name",
            "All role preferences",
        ]
        
        # Add individual rank columns
        for role in role_names:
            role_short = role.split()[0]
            col_name = f"{role_short} Rank"
            if col_name in filtered_by_role.columns:
                display_cols.append(col_name)
        
        # Add other useful columns
        other_cols = [
            cfg["review"]["status_field"],
            cfg["review"]["notes_field"],
            cfg["review"]["ai_flag_field"],
            "gpa_flag"
        ]
        display_cols.extend([col for col in other_cols if col in filtered_by_role.columns])
        
        st.write(f"### Results: {len(filtered_by_role)} applications match filters")
        st.dataframe(filtered_by_role[display_cols], width="stretch", hide_index=True)

        # Download button for filtered results
        csv_export = filtered_by_role[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            f"Download filtered results",
            data=csv_export,
            file_name=f"position_preferences_filtered.csv",
            mime="text/csv",
        )
    else:
        st.info("No applications match the selected filters.")
        
    # Add summary statistics about position preferences
    st.subheader("Position Preference Statistics")
    
    stats_cols = st.columns(len(role_names))
    for i, role in enumerate(role_names):
        role_short = role.split()[0]
        col_name = f"{role_short} Rank"
        
        with stats_cols[i]:
            st.write(f"**{role}**")
            
            if col_name in filtered_df.columns:
                # Count preferences
                rank_1 = len(filtered_df[filtered_df[col_name] == "1"])
                rank_2 = len(filtered_df[filtered_df[col_name] == "2"])
                rank_3 = len(filtered_df[filtered_df[col_name] == "3"])
                total_ranked = rank_1 + rank_2 + rank_3
                
                # Display metrics
                st.metric("Ranked #1", rank_1)
                st.metric("Ranked #2", rank_2)
                st.metric("Ranked #3", rank_3)
                st.metric("Total", total_ranked)
                
                # Create a mini chart
                if total_ranked > 0:
                    chart_data = {
                        "Rank": ["#1", "#2", "#3"],
                        "Count": [rank_1, rank_2, rank_3]
                    }
                    chart_df = pd.DataFrame(chart_data)
                    st.bar_chart(chart_df, x="Rank", y="Count")

    # Add role-specific tabs for detailed analysis
    st.subheader("Position-Specific Analysis")
    role_tabs = st.tabs(role_names)
    
    for i, role in enumerate(role_names):
        with role_tabs[i]:
            st.write(f"### {role} Preference Analysis")
            
            # Get applicants who ranked this role
            role_short = role.split()[0]
            rank_col = f"{role_short} Rank"
            
            if rank_col in filtered_df.columns:
                ranked_df = filtered_df[filtered_df[rank_col].isin(["1", "2", "3"])].copy()
                
                if not ranked_df.empty:
                    # Sort by rank
                    ranked_df = ranked_df.sort_values(by=rank_col)
                    
                    # Display statistics
                    rank_counts = ranked_df[rank_col].value_counts().sort_index()
                    
                    st.write(f"**Total applicants who ranked this position:** {len(ranked_df)}")
                    
                    # Create columns for the rankings
                    rank_cols = st.columns(3)
                    for j, rank in enumerate(["1", "2", "3"]):
                        count = rank_counts.get(rank, 0)
                        rank_cols[j].metric(f"Rank #{rank}", count)
                    
                    # Show the ranked applicants
                    st.write("#### Applicants who ranked this position")
                    
                    # Add sorting options
                    sort_option = st.radio(
                        f"Sort {role} applicants by:",
                        ["Preference Rank", "Name", "Review Status"],
                        horizontal=True,
                        key=f"sort_{role}_tab"
                    )
                    
                    if sort_option == "Preference Rank":
                        pass  # Already sorted by rank
                    elif sort_option == "Name":
                        # Sort by Last Name, First Name
                        if "Last Name" in ranked_df.columns and "First Name" in ranked_df.columns:
                            ranked_df = ranked_df.sort_values(by=["Last Name", "First Name"])
                    elif sort_option == "Review Status":
                        # Sort by review status
                        status_col = cfg["review"]["status_field"]
                        if status_col in ranked_df.columns:
                            ranked_df = ranked_df.sort_values(by=status_col)
                    
                    # Display columns for the table
                    display_cols = [
                        unique_key,
                        "Username",
                        "First Name", 
                        "Last Name",
                        rank_col,
                        cfg["review"]["status_field"],
                        cfg["review"]["notes_field"],
                    ]
                    
                    display_cols = [col for col in display_cols if col in ranked_df.columns]
                    
                    st.dataframe(ranked_df[display_cols], width="stretch", hide_index=True)
                    
                    # Download button
                    csv_export = ranked_df[display_cols].to_csv(index=False).encode("utf-8")
                    st.download_button(
                        f"Download {role} ranked applicants",
                        data=csv_export,
                        file_name=f"{role.lower().replace(' ', '_')}_ranked.csv",
                        mime="text/csv",
                    )
                else:
                    st.info(f"No applicants have ranked {role} yet.")
            else:
                st.info(f"No ranking data available for {role}.")